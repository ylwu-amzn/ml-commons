/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.deploy;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.common.CommonValue.ML_MODEL_INDEX;
import static org.opensearch.ml.settings.MLCommonsSettings.ML_COMMONS_VALIDATE_BACKEND_ROLES;
import static org.opensearch.ml.utils.MLExceptionUtils.logException;
import static org.opensearch.ml.utils.MLNodeUtils.createXContentParserFromRegistry;
import static org.opensearch.ml.utils.RestActionUtils.getFetchSourceContext;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionListenerResponseHandler;
import org.opensearch.action.FailedNodeException;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.nodes.TransportNodesAction;
import org.opensearch.client.Client;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.node.DiscoveryNodes;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.commons.authuser.User;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.breaker.MLCircuitBreakerService;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.exception.MLResourceNotFoundException;
import org.opensearch.ml.common.exception.MLValidationException;
import org.opensearch.ml.common.transport.deploy.*;
import org.opensearch.ml.common.transport.forward.*;
import org.opensearch.ml.common.transport.model.MLModelGetRequest;
import org.opensearch.ml.engine.ModelHelper;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.stats.MLStats;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.ml.utils.MLExceptionUtils;
import org.opensearch.ml.utils.RestActionUtils;
import org.opensearch.ml.utils.SecurityUtils;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

@Log4j2
public class TransportDeployModelOnNodeAction extends
    TransportNodesAction<MLDeployModelNodesRequest, MLDeployModelNodesResponse, MLDeployModelNodeRequest, MLDeployModelNodeResponse> {
    TransportService transportService;
    ModelHelper modelHelper;
    MLTaskManager mlTaskManager;
    MLModelManager mlModelManager;
    ClusterService clusterService;
    ThreadPool threadPool;
    Client client;
    NamedXContentRegistry xContentRegistry;
    MLCircuitBreakerService mlCircuitBreakerService;
    MLStats mlStats;

    private volatile boolean filterByEnabled;

    @Inject
    public TransportDeployModelOnNodeAction(
        TransportService transportService,
        ActionFilters actionFilters,
        ModelHelper modelHelper,
        MLTaskManager mlTaskManager,
        MLModelManager mlModelManager,
        ClusterService clusterService,
        ThreadPool threadPool,
        Client client,
        NamedXContentRegistry xContentRegistry,
        MLCircuitBreakerService mlCircuitBreakerService,
        MLStats mlStats,
        Settings settings
    ) {
        super(
            MLDeployModelOnNodeAction.NAME,
            threadPool,
            clusterService,
            transportService,
            actionFilters,
            MLDeployModelNodesRequest::new,
            MLDeployModelNodeRequest::new,
            ThreadPool.Names.MANAGEMENT,
            MLDeployModelNodeResponse.class
        );
        this.transportService = transportService;
        this.modelHelper = modelHelper;
        this.mlTaskManager = mlTaskManager;
        this.mlModelManager = mlModelManager;
        this.clusterService = clusterService;
        this.threadPool = threadPool;
        this.client = client;
        this.xContentRegistry = xContentRegistry;
        this.mlCircuitBreakerService = mlCircuitBreakerService;
        this.mlStats = mlStats;
        filterByEnabled = ML_COMMONS_VALIDATE_BACKEND_ROLES.get(settings);
        clusterService.getClusterSettings().addSettingsUpdateConsumer(ML_COMMONS_VALIDATE_BACKEND_ROLES, it -> filterByEnabled = it);
    }

    @Override
    protected MLDeployModelNodesResponse newResponse(
        MLDeployModelNodesRequest nodesRequest,
        List<MLDeployModelNodeResponse> responses,
        List<FailedNodeException> failures
    ) {
        return new MLDeployModelNodesResponse(clusterService.getClusterName(), responses, failures);
    }

    @Override
    protected MLDeployModelNodeRequest newNodeRequest(MLDeployModelNodesRequest request) {
        return new MLDeployModelNodeRequest(request);
    }

    @Override
    protected MLDeployModelNodeResponse newNodeResponse(StreamInput in) throws IOException {
        return new MLDeployModelNodeResponse(in);
    }

    @Override
    protected MLDeployModelNodeResponse nodeOperation(MLDeployModelNodeRequest request) {
        return createDeployModelNodeResponse(request.getMLDeployModelNodesRequest());
    }

    private MLDeployModelNodeResponse createDeployModelNodeResponse(MLDeployModelNodesRequest MLDeployModelNodesRequest) {
        MLDeployModelInput deployModelInput = MLDeployModelNodesRequest.getMlDeployModelInput();
        String modelId = deployModelInput.getModelId();
        MLModelGetRequest mlModelGetRequest = new MLModelGetRequest(modelId, false);
        FetchSourceContext fetchSourceContext = getFetchSourceContext(mlModelGetRequest.isReturnContent());
        GetRequest getRequest = new GetRequest(ML_MODEL_INDEX).id(modelId).fetchSourceContext(fetchSourceContext);
        User user = RestActionUtils.getUserContext(client);

        try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
            client.get(getRequest, ActionListener.wrap(r -> {
                if (r != null && r.isExists()) {
                    try (XContentParser parser = createXContentParserFromRegistry(xContentRegistry, r.getSourceAsBytesRef())) {
                        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
                        MLModel mlModel = MLModel.parse(parser);
                        log.info(mlModel.getModelGroupId());

                        if ((mlModel.getModelGroupId() != null)
                            && (filterByEnabled)
                            && (!SecurityUtils.validateModelGroupAccess(user, mlModel.getModelGroupId(), client))) {
                            log.info("backend roles did not match");
                            log.error("User doesn't have valid privilege to perform this operation");
                            throw new MLValidationException("User doesn't have valid privilege to perform this operation");
                        }
                    }
                }
            }, e -> { throw new MLResourceNotFoundException("Fail to find model"); }));
        } catch (Exception e) {
            throw e;
        }

        String taskId = deployModelInput.getTaskId();
        Integer nodeCount = deployModelInput.getNodeCount();
        String coordinatingNodeId = deployModelInput.getCoordinatingNodeId();
        MLTask mlTask = deployModelInput.getMlTask();
        String modelContentHash = deployModelInput.getModelContentHash();

        Map<String, String> modelDeployStatus = new HashMap<>();
        modelDeployStatus.put(modelId, "received");

        String localNodeId = clusterService.localNode().getId();

        ActionListener<MLForwardResponse> taskDoneListener = ActionListener
            .wrap(
                res -> { log.info("deploy model task done " + taskId); },
                ex -> { logException("Deploy model task failed: " + taskId, ex, log); }
            );

        deployModel(modelId, modelContentHash, mlTask.getFunctionName(), localNodeId, coordinatingNodeId, mlTask, ActionListener.wrap(r -> {
            MLForwardInput mlForwardInput = MLForwardInput
                .builder()
                .requestType(MLForwardRequestType.DEPLOY_MODEL_DONE)
                .taskId(taskId)
                .modelId(modelId)
                .workerNodeId(clusterService.localNode().getId())
                .build();
            MLForwardRequest deployModelDoneMessage = new MLForwardRequest(mlForwardInput);

            transportService
                .sendRequest(
                    getNodeById(coordinatingNodeId),
                    MLForwardAction.NAME,
                    deployModelDoneMessage,
                    new ActionListenerResponseHandler<MLForwardResponse>(taskDoneListener, MLForwardResponse::new)
                );
        }, e -> {
            MLForwardInput mlForwardInput = MLForwardInput
                .builder()
                .requestType(MLForwardRequestType.DEPLOY_MODEL_DONE)
                .taskId(taskId)
                .modelId(modelId)
                .workerNodeId(clusterService.localNode().getId())
                .error(MLExceptionUtils.getRootCauseMessage(e))
                .build();
            MLForwardRequest deployModelDoneMessage = new MLForwardRequest(mlForwardInput);

            transportService
                .sendRequest(
                    getNodeById(coordinatingNodeId),
                    MLForwardAction.NAME,
                    deployModelDoneMessage,
                    new ActionListenerResponseHandler<MLForwardResponse>(taskDoneListener, MLForwardResponse::new)
                );
        }));

        return new MLDeployModelNodeResponse(clusterService.localNode(), modelDeployStatus);
    }

    private DiscoveryNode getNodeById(String nodeId) {
        DiscoveryNodes nodes = clusterService.state().getNodes();
        Iterator<DiscoveryNode> iterator = nodes.iterator();
        while (iterator.hasNext()) {
            DiscoveryNode node = iterator.next();
            if (node.getId().equals(nodeId)) {
                return node;
            }
        }
        return null;
    }

    private void deployModel(
        String modelId,
        String modelContentHash,
        FunctionName functionName,
        String localNodeId,
        String coordinatingNodeId,
        MLTask mlTask,
        ActionListener<String> listener
    ) {
        try {
            log.debug("start deploying model {}", modelId);
            mlModelManager.deployModel(modelId, modelContentHash, functionName, mlTask, ActionListener.runBefore(listener, () -> {
                if (!coordinatingNodeId.equals(localNodeId)) {
                    mlTaskManager.remove(mlTask.getTaskId());
                }
            }));
        } catch (Exception e) {
            logException("Failed to deploy model " + modelId, e, log);
            listener.onFailure(e);
        }
    }

}
