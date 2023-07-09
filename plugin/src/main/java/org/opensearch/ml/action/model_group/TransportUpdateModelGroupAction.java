/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.model_group;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.common.CommonValue.ML_MODEL_GROUP_INDEX;
import static org.opensearch.ml.utils.MLExceptionUtils.logException;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.action.update.UpdateRequest;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.util.CollectionUtils;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.commons.authuser.User;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.AccessMode;
import org.opensearch.ml.common.MLModelGroup;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.exception.MLResourceNotFoundException;
import org.opensearch.ml.common.transport.model_group.MLUpdateModelGroupAction;
import org.opensearch.ml.common.transport.model_group.MLUpdateModelGroupInput;
import org.opensearch.ml.common.transport.model_group.MLUpdateModelGroupRequest;
import org.opensearch.ml.common.transport.model_group.MLUpdateModelGroupResponse;
import org.opensearch.ml.helper.ModelAccessControlHelper;
import org.opensearch.ml.utils.MLNodeUtils;
import org.opensearch.ml.utils.RestActionUtils;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

import com.google.common.collect.ImmutableList;

import lombok.extern.log4j.Log4j2;

@Log4j2
public class TransportUpdateModelGroupAction extends HandledTransportAction<ActionRequest, MLUpdateModelGroupResponse> {

    private final TransportService transportService;
    private final ActionFilters actionFilters;
    private Client client;
    private NamedXContentRegistry xContentRegistry;
    ClusterService clusterService;

    ModelAccessControlHelper modelAccessControlHelper;

    @Inject
    public TransportUpdateModelGroupAction(
        TransportService transportService,
        ActionFilters actionFilters,
        Client client,
        NamedXContentRegistry xContentRegistry,
        ClusterService clusterService,
        ModelAccessControlHelper modelAccessControlHelper
    ) {
        super(MLUpdateModelGroupAction.NAME, transportService, actionFilters, MLUpdateModelGroupRequest::new);
        this.actionFilters = actionFilters;
        this.transportService = transportService;
        this.client = client;
        this.xContentRegistry = xContentRegistry;
        this.clusterService = clusterService;
        this.modelAccessControlHelper = modelAccessControlHelper;
    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<MLUpdateModelGroupResponse> listener) {
        MLUpdateModelGroupRequest updateModelGroupRequest = MLUpdateModelGroupRequest.fromActionRequest(request);
        MLUpdateModelGroupInput updateModelGroupInput = updateModelGroupRequest.getUpdateModelGroupInput();
        String modelGroupId = updateModelGroupInput.getModelGroupID();
        User user = RestActionUtils.getUserContext(client);
        if (modelAccessControlHelper.isSecurityEnabledAndModelAccessControlEnabled(user)) {
            GetRequest getModelGroupRequest = new GetRequest(ML_MODEL_GROUP_INDEX).id(modelGroupId);
            try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                client.get(getModelGroupRequest, ActionListener.wrap(modelGroup -> {
                    if (modelGroup.isExists()) {
                        try (
                            XContentParser parser = MLNodeUtils
                                .createXContentParserFromRegistry(NamedXContentRegistry.EMPTY, modelGroup.getSourceAsBytesRef())
                        ) {
                            ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
                            MLModelGroup mlModelGroup = MLModelGroup.parse(parser);
                            validateRequestForAccessControl(updateModelGroupInput, user, mlModelGroup);
                            updateModelGroup(modelGroupId, modelGroup.getSource(), updateModelGroupInput, listener, user);
                        }
                    } else {
                        listener.onFailure(new MLResourceNotFoundException("Failed to find model group"));
                    }
                }, e -> {
                    logException("Failed to get model group", e, log);
                    listener.onFailure(e);
                }));
            } catch (Exception e) {
                logException("Failed to Update model group", e, log);
                listener.onFailure(e);
            }
        } else {
            validateSecurityDisabledOrModelAccessControlDisabled(updateModelGroupInput);
            updateModelGroup(modelGroupId, new HashMap<>(), updateModelGroupInput, listener, user);
        }
    }

    private void updateModelGroup(
        String modelGroupId,
        Map<String, Object> source,
        MLUpdateModelGroupInput updateModelGroupInput,
        ActionListener<MLUpdateModelGroupResponse> listener,
        User user
    ) {
        if (updateModelGroupInput.getModelAccessMode() != null) {
            source.put(MLModelGroup.ACCESS, updateModelGroupInput.getModelAccessMode().getValue());
            if (AccessMode.RESTRICTED != updateModelGroupInput.getModelAccessMode()) {
                source.put(MLModelGroup.BACKEND_ROLES_FIELD, ImmutableList.of());
            }
        } else if (updateModelGroupInput.getBackendRoles() != null
            || Boolean.TRUE.equals(updateModelGroupInput.getIsAddAllBackendRoles())) {
            source.put(MLModelGroup.ACCESS, AccessMode.RESTRICTED.getValue());
        }
        if (updateModelGroupInput.getBackendRoles() != null) {
            source.put(MLModelGroup.BACKEND_ROLES_FIELD, updateModelGroupInput.getBackendRoles());
        }
        if (Boolean.TRUE.equals(updateModelGroupInput.getIsAddAllBackendRoles())) {
            source.put(MLModelGroup.BACKEND_ROLES_FIELD, user.getBackendRoles());
        }
        if (StringUtils.isNotBlank(updateModelGroupInput.getName())) {
            source.put(MLModelGroup.MODEL_GROUP_NAME_FIELD, updateModelGroupInput.getName());
        }
        if (StringUtils.isNotBlank(updateModelGroupInput.getDescription())) {
            source.put(MLModelGroup.DESCRIPTION_FIELD, updateModelGroupInput.getDescription());
        }

        UpdateRequest updateModelGroupRequest = new UpdateRequest();
        updateModelGroupRequest.index(ML_MODEL_GROUP_INDEX).id(modelGroupId).doc(source);
        try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
            client
                .update(
                    updateModelGroupRequest,
                    ActionListener.wrap(r -> { listener.onResponse(new MLUpdateModelGroupResponse("Updated")); }, e -> {
                        log.error("Failed to update Model Group", e);
                        throw new MLException("Failed to update Model Group", e);
                    })
                );
        } catch (Exception e) {
            logException("Failed to Update model group ", e, log);
            listener.onFailure(e);
        }
    }

    private void validateRequestForAccessControl(MLUpdateModelGroupInput input, User user, MLModelGroup mlModelGroup) {
        if (hasAccessControlChange(input)) {
            if (!modelAccessControlHelper.isOwner(mlModelGroup.getOwner(), user) && !modelAccessControlHelper.isAdmin(user)) {
                throw new IllegalArgumentException("Only owner or admin can update access control data.");
            } else if (modelAccessControlHelper.isOwner(mlModelGroup.getOwner(), user)
                && !modelAccessControlHelper.isAdmin(user)
                && !modelAccessControlHelper.isOwnerStillHasPermission(user, mlModelGroup)) {
                throw new IllegalArgumentException(
                    "You don’t have the specified backend role to update access control data. For more information, contact your administrator."
                );
            }
        }
        if (!modelAccessControlHelper.isAdmin(user)
            && !modelAccessControlHelper.isOwner(mlModelGroup.getOwner(), user)
            && !modelAccessControlHelper.isUserHasBackendRole(user, mlModelGroup)) {
            throw new IllegalArgumentException("You don't have permissions to perform this operation on this model group.");
        }
        AccessMode modelAccessMode = input.getModelAccessMode();
        if ((AccessMode.PUBLIC == modelAccessMode || AccessMode.PRIVATE == modelAccessMode)
            && (!CollectionUtils.isEmpty(input.getBackendRoles()) || Boolean.TRUE.equals(input.getIsAddAllBackendRoles()))) {
            throw new IllegalArgumentException("You can specify backend roles only for a model group with the restricted access mode.");
        } else if (modelAccessMode == null || AccessMode.RESTRICTED == modelAccessMode) {
            if (modelAccessControlHelper.isAdmin(user) && Boolean.TRUE.equals(input.getIsAddAllBackendRoles())) {
                throw new IllegalArgumentException("Admin users cannot add all backend roles to a model group.");
            }
            if (Boolean.TRUE.equals(input.getIsAddAllBackendRoles()) && CollectionUtils.isEmpty(user.getBackendRoles())) {
                throw new IllegalArgumentException("You don’t have any backend roles.");
            }
            if (CollectionUtils.isEmpty(input.getBackendRoles()) && Boolean.FALSE.equals(input.getIsAddAllBackendRoles())) {
                throw new IllegalArgumentException("User have to specify backend roles when add all backend roles is set to false.");
            }
            if (!CollectionUtils.isEmpty(input.getBackendRoles()) && Boolean.TRUE.equals(input.getIsAddAllBackendRoles())) {
                throw new IllegalArgumentException("You cannot specify backend roles and add all backend roles at the same time.");
            }
            if (AccessMode.RESTRICTED == modelAccessMode
                && CollectionUtils.isEmpty(input.getBackendRoles())
                && !Boolean.TRUE.equals(input.getIsAddAllBackendRoles())) {
                throw new IllegalArgumentException(
                    "You must specify one or more backend roles or add all backend roles to register a restricted model group."
                );
            }
            if (!modelAccessControlHelper.isAdmin(user)
                && !CollectionUtils.isEmpty(input.getBackendRoles())
                && !new HashSet<>(user.getBackendRoles()).containsAll(input.getBackendRoles())) {
                throw new IllegalArgumentException("You don't have the backend roles specified.");
            }
        }
    }

    private boolean hasAccessControlChange(MLUpdateModelGroupInput input) {
        return input.getModelAccessMode() != null || input.getIsAddAllBackendRoles() != null || input.getBackendRoles() != null;
    }

    private void validateSecurityDisabledOrModelAccessControlDisabled(MLUpdateModelGroupInput input) {
        if (input.getModelAccessMode() != null || input.getIsAddAllBackendRoles() != null || input.getBackendRoles() != null) {
            throw new IllegalArgumentException(
                "You cannot specify model access control parameters because the Security plugin or model access control is disabled on your cluster."
            );
        }
    }

}
