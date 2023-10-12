/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.models;

import static org.opensearch.core.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.common.CommonValue.ML_MODEL_INDEX;
import static org.opensearch.ml.common.MLModel.ALGORITHM_FIELD;
import static org.opensearch.ml.utils.MLNodeUtils.createXContentParserFromRegistry;
import static org.opensearch.ml.utils.RestActionUtils.getFetchSourceContext;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.get.GetResponse;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.commons.authuser.User;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.IndexNotFoundException;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.connector.Connector;
import org.opensearch.ml.common.exception.MLResourceNotFoundException;
import org.opensearch.ml.common.exception.MLValidationException;
import org.opensearch.ml.common.transport.model.MLModelGetAction;
import org.opensearch.ml.common.transport.model.MLModelGetRequest;
import org.opensearch.ml.common.transport.model.MLModelGetResponse;
import org.opensearch.ml.helper.ModelAccessControlHelper;
import org.opensearch.ml.utils.RestActionUtils;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.tasks.Task;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;
import lombok.extern.log4j.Log4j2;

import javax.naming.InvalidNameException;
import javax.naming.ldap.LdapName;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@Log4j2
@FieldDefaults(level = AccessLevel.PRIVATE)
public class GetModelTransportAction extends HandledTransportAction<ActionRequest, MLModelGetResponse> {
    public static final String SECURITY_AUTHCZ_ADMIN_DN = "plugins.security.authcz.admin_dn";
    public static final String OPENDISTRO_SECURITY_CONFIG_PREFIX = "_opendistro_security_";
    public static final String OPENDISTRO_SECURITY_SSL_PRINCIPAL = OPENDISTRO_SECURITY_CONFIG_PREFIX + "ssl_principal";
    Client client;
    NamedXContentRegistry xContentRegistry;
    ClusterService clusterService;

    ModelAccessControlHelper modelAccessControlHelper;
    final Set<LdapName> adminDn = new HashSet<LdapName>();
    ThreadPool threadPool;

    @Inject
    public GetModelTransportAction(
        TransportService transportService,
        ActionFilters actionFilters,
        Client client,
        NamedXContentRegistry xContentRegistry,
        ClusterService clusterService,
        ModelAccessControlHelper modelAccessControlHelper,
        ThreadPool threadPool
    ) {
        super(MLModelGetAction.NAME, transportService, actionFilters, MLModelGetRequest::new);
        this.client = client;
        this.xContentRegistry = xContentRegistry;
        this.clusterService = clusterService;
        this.modelAccessControlHelper = modelAccessControlHelper;
        this.threadPool = threadPool;
    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<MLModelGetResponse> actionListener) {
        MLModelGetRequest mlModelGetRequest = MLModelGetRequest.fromActionRequest(request);
        String modelId = mlModelGetRequest.getModelId();
        FetchSourceContext fetchSourceContext = getFetchSourceContext(mlModelGetRequest.isReturnContent());
        GetRequest getRequest = new GetRequest(ML_MODEL_INDEX).id(modelId).fetchSourceContext(fetchSourceContext);
        User user = RestActionUtils.getUserContext(client);

        if (!isAdminDN()) {
            actionListener.onFailure(new OpenSearchStatusException("You don't have permission to access this model", RestStatus.FORBIDDEN));
        }

        try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
            ActionListener<MLModelGetResponse> wrappedListener = ActionListener.runBefore(actionListener, () -> context.restore());
            client.get(getRequest, ActionListener.wrap(r -> {
                if (r != null && r.isExists()) {
                    try (XContentParser parser = createXContentParserFromRegistry(xContentRegistry, r.getSourceAsBytesRef())) {
                        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
                        GetResponse getResponse = r;
                        String algorithmName = getResponse.getSource().get(ALGORITHM_FIELD).toString();

                        MLModel mlModel = MLModel.parse(parser, algorithmName);
                        modelAccessControlHelper
                            .validateModelGroupAccess(user, mlModel.getModelGroupId(), client, ActionListener.wrap(access -> {
                                if (!access) {
                                    wrappedListener
                                        .onFailure(
                                            new MLValidationException("User Doesn't have privilege to perform this operation on this model")
                                        );
                                } else {
                                    log.debug("Completed Get Model Request, id:{}", modelId);
                                    Connector connector = mlModel.getConnector();
                                    if (connector != null) {
                                        connector.removeCredential();
                                    }
                                    wrappedListener.onResponse(MLModelGetResponse.builder().mlModel(mlModel).build());
                                }
                            }, e -> {
                                log.error("Failed to validate Access for Model Id " + modelId, e);
                                wrappedListener.onFailure(e);
                            }));

                    } catch (Exception e) {
                        log.error("Failed to parse ml model " + r.getId(), e);
                        wrappedListener.onFailure(e);
                    }
                } else {
                    wrappedListener
                        .onFailure(
                            new OpenSearchStatusException(
                                "Failed to find model with the provided model id: " + modelId,
                                RestStatus.NOT_FOUND
                            )
                        );
                }
            }, e -> {
                if (e instanceof IndexNotFoundException) {
                    wrappedListener.onFailure(new MLResourceNotFoundException("Fail to find model"));
                } else {
                    log.error("Failed to get ML model " + modelId, e);
                    wrappedListener.onFailure(e);
                }
            }));
        } catch (Exception e) {
            log.error("Failed to get ML model " + modelId, e);
            actionListener.onFailure(e);
        }

    }

    public boolean isAdminDN() {
        final List<String> adminDnsA = clusterService.getSettings().getAsList(SECURITY_AUTHCZ_ADMIN_DN, Collections.emptyList());

        for (String dn : adminDnsA) {
            try {
                log.debug("{} is registered as an admin dn", dn);
                adminDn.add(new LdapName(dn));
            } catch (final InvalidNameException e) {
                log.error("Unable to parse admin dn {}", dn, e);
            }
        }

        ThreadContext threadContext = this.threadPool.getThreadContext();
        final String sslPrincipal = (String) threadContext.getTransient(OPENDISTRO_SECURITY_SSL_PRINCIPAL);
        boolean adminDN = isAdminDN(sslPrincipal);
        return adminDN;
    }

    public boolean isAdminDN(String dn) {

        if (dn == null) return false;

        try {
            return isAdminDN(new LdapName(dn));
        } catch (InvalidNameException e) {
            return false;
        }
    }

    private boolean isAdminDN(LdapName dn) {
        if (dn == null) return false;

        boolean isAdmin = adminDn.contains(dn);

        if (log.isTraceEnabled()) {
            log.trace("Is principal {} an admin cert? {}", dn.toString(), isAdmin);
        }

        return isAdmin;
    }
}
