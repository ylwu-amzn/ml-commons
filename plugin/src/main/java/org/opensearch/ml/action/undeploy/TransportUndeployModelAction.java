/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.undeploy;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.common.CommonValue.*;
import static org.opensearch.ml.common.MLModel.MODEL_STATE_FIELD;
import static org.opensearch.ml.settings.MLCommonsSettings.ML_COMMONS_VALIDATE_BACKEND_ROLES;
import static org.opensearch.ml.utils.MLNodeUtils.createXContentParserFromRegistry;
import static org.opensearch.ml.utils.RestActionUtils.getFetchSourceContext;

import java.io.IOException;
import java.util.*;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.action.FailedNodeException;
import org.opensearch.action.bulk.BulkRequest;
import org.opensearch.action.bulk.BulkResponse;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.nodes.TransportNodesAction;
import org.opensearch.action.update.UpdateRequest;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.commons.authuser.User;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.cluster.DiscoveryNodeHelper;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.common.transport.model.MLModelGetRequest;
import org.opensearch.ml.common.transport.sync.MLSyncUpAction;
import org.opensearch.ml.common.transport.sync.MLSyncUpInput;
import org.opensearch.ml.common.transport.sync.MLSyncUpNodesRequest;
import org.opensearch.ml.common.transport.undeploy.*;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.stats.MLNodeLevelStat;
import org.opensearch.ml.stats.MLStats;
import org.opensearch.ml.utils.RestActionUtils;
import org.opensearch.ml.utils.SecurityUtils;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import com.google.common.collect.ImmutableMap;

@Log4j2
public class TransportUndeployModelAction extends
    TransportNodesAction<MLUndeployModelNodesRequest, MLUndeployModelNodesResponse, MLUndeployModelNodeRequest, MLUndeployModelNodeResponse> {
    private final MLModelManager mlModelManager;
    private final ClusterService clusterService;
    private final Client client;
    private DiscoveryNodeHelper nodeFilter;
    private final MLStats mlStats;
    NamedXContentRegistry xContentRegistry;

    private volatile boolean filterByEnabled;

    @Inject
    public TransportUndeployModelAction(
        TransportService transportService,
        ActionFilters actionFilters,
        MLModelManager mlModelManager,
        ClusterService clusterService,
        ThreadPool threadPool,
        Client client,
        DiscoveryNodeHelper nodeFilter,
        MLStats mlStats,
        NamedXContentRegistry xContentRegistry,
        Settings settings
    ) {
        super(
            MLUndeployModelAction.NAME,
            threadPool,
            clusterService,
            transportService,
            actionFilters,
            MLUndeployModelNodesRequest::new,
            MLUndeployModelNodeRequest::new,
            ThreadPool.Names.MANAGEMENT,
            MLUndeployModelNodeResponse.class
        );
        this.mlModelManager = mlModelManager;
        this.clusterService = clusterService;
        this.client = client;
        this.nodeFilter = nodeFilter;
        this.mlStats = mlStats;
        this.xContentRegistry = xContentRegistry;
        filterByEnabled = ML_COMMONS_VALIDATE_BACKEND_ROLES.get(settings);
        clusterService.getClusterSettings().addSettingsUpdateConsumer(ML_COMMONS_VALIDATE_BACKEND_ROLES, it -> filterByEnabled = it);
    }

    @Override
    protected MLUndeployModelNodesResponse newResponse(
        MLUndeployModelNodesRequest nodesRequest,
        List<MLUndeployModelNodeResponse> responses,
        List<FailedNodeException> failures
    ) {
        if (responses != null) {
            Map<String, List<String>> removedNodeMap = new HashMap<>();
            Map<String, Integer> modelWorkNodeCounts = new HashMap<>();
            Set<String> invalidAccessModels = new HashSet<>();
            User user = RestActionUtils.getUserContext(client);
            responses.stream().forEach(r -> {
                Set<String> notFoundModels = new HashSet<>();
                Map<String, Integer> nodeCounts = r.getModelWorkerNodeCounts();
                if (nodeCounts != null) {
                    for (Map.Entry<String, Integer> entry : nodeCounts.entrySet()) {
                        if (!modelWorkNodeCounts.containsKey(entry.getKey())
                            || modelWorkNodeCounts.get(entry.getKey()) < entry.getValue()) {
                            modelWorkNodeCounts.put(entry.getKey(), entry.getValue());
                        }
                    }
                }

                Map<String, String> modelUndeployStatus = r.getModelUndeployStatus();
                for (Map.Entry<String, String> entry : modelUndeployStatus.entrySet()) {
                    String status = entry.getValue();
                    String modelId = entry.getKey();

                    MLModelGetRequest mlModelGetRequest = new MLModelGetRequest(modelId, false);
                    FetchSourceContext fetchSourceContext = getFetchSourceContext(mlModelGetRequest.isReturnContent());
                    GetRequest getRequest = new GetRequest(ML_MODEL_INDEX).id(modelId).fetchSourceContext(fetchSourceContext);

                    try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                        client.get(getRequest, ActionListener.wrap(model -> {
                            if (model != null && model.isExists()) {
                                try (
                                    XContentParser parser = createXContentParserFromRegistry(xContentRegistry, model.getSourceAsBytesRef())
                                ) {
                                    ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
                                    MLModel mlModel = MLModel.parse(parser);

                                    if (filterByEnabled
                                        && !SecurityUtils.validateModelGroupAccess(user, mlModel.getModelGroupId(), client)) {
                                        log.error("User doesn't have valid privilege to perform this operation");
                                        invalidAccessModels.add(modelId);
                                    }
                                }
                            }
                        }, e -> {
                            log.error("Fail to find model");
                            invalidAccessModels.add(modelId);
                        }));
                    } catch (Exception e) {
                        throw e;
                    }

                    if (UNDEPLOYED.equals(status) || NOT_FOUND.equals(status)) {
                        if (!removedNodeMap.containsKey(modelId)) {
                            removedNodeMap.put(modelId, new ArrayList<>());
                        }
                        removedNodeMap.get(modelId).add(r.getNode().getId());
                    }
                    if (NOT_FOUND.equals(status)) {
                        notFoundModels.add(entry.getKey());
                    }
                }
                notFoundModels.forEach(m -> modelUndeployStatus.remove(m));
            });
            Map<String, String[]> removedNodes = new HashMap<>();
            for (Map.Entry<String, List<String>> entry : removedNodeMap.entrySet()) {
                removedNodes.put(entry.getKey(), entry.getValue().toArray(new String[0]));
                log.debug("removed node for model: {}, {}", entry.getKey(), Arrays.toString(entry.getValue().toArray(new String[0])));
            }
            MLSyncUpInput syncUpInput = MLSyncUpInput.builder().removedWorkerNodes(removedNodes).build();

            MLSyncUpNodesRequest syncUpRequest = new MLSyncUpNodesRequest(nodeFilter.getAllNodes(), syncUpInput);
            try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                if (removedNodeMap.size() > 0) {
                    BulkRequest bulkRequest = new BulkRequest();
                    for (String modelId : removedNodeMap.keySet()) {
                        if (!invalidAccessModels.contains(modelId)) {
                            UpdateRequest updateRequest = new UpdateRequest();
                            int removedNodeCount = removedNodeMap.get(modelId).size();
                            MLModelState mlModelState = modelWorkNodeCounts.get(modelId) > removedNodeCount
                                ? MLModelState.PARTIALLY_DEPLOYED
                                : MLModelState.UNDEPLOYED;
                            updateRequest.index(ML_MODEL_INDEX).id(modelId).doc(ImmutableMap.of(MODEL_STATE_FIELD, mlModelState));
                            bulkRequest.add(updateRequest);
                        }
                    }
                    ActionListener<BulkResponse> actionListenr = ActionListener
                        .wrap(
                            r -> {
                                log
                                    .debug(
                                        "updated model state as undeployed for : {}",
                                        Arrays.toString(removedNodeMap.keySet().toArray(new String[0]))
                                    );
                            },
                            e -> { log.error("Failed to update model state as undeployed", e); }
                        );
                    client.bulk(bulkRequest, ActionListener.runAfter(actionListenr, () -> { syncUpUndeployedModels(syncUpRequest); }));
                } else {
                    syncUpUndeployedModels(syncUpRequest);
                }
            }
        }
        return new MLUndeployModelNodesResponse(clusterService.getClusterName(), responses, failures);
    }

    private void syncUpUndeployedModels(MLSyncUpNodesRequest syncUpRequest) {
        client
            .execute(
                MLSyncUpAction.INSTANCE,
                syncUpRequest,
                ActionListener
                    .wrap(r -> log.debug("sync up removed nodes successfully"), e -> log.error("failed to sync up removed node", e))
            );
    }

    @Override
    protected MLUndeployModelNodeRequest newNodeRequest(MLUndeployModelNodesRequest request) {
        return new MLUndeployModelNodeRequest(request);
    }

    @Override
    protected MLUndeployModelNodeResponse newNodeResponse(StreamInput in) throws IOException {
        return new MLUndeployModelNodeResponse(in);
    }

    @Override
    protected MLUndeployModelNodeResponse nodeOperation(MLUndeployModelNodeRequest request) {
        return createUndeployModelNodeResponse(request.getMlUndeployModelNodesRequest());
    }

    private MLUndeployModelNodeResponse createUndeployModelNodeResponse(MLUndeployModelNodesRequest MLUndeployModelNodesRequest) {
        mlStats.getStat(MLNodeLevelStat.ML_NODE_EXECUTING_TASK_COUNT).increment();
        mlStats.getStat(MLNodeLevelStat.ML_NODE_TOTAL_REQUEST_COUNT).increment();

        String[] modelIds = MLUndeployModelNodesRequest.getModelIds();

        Map<String, Integer> modelWorkerNodeCounts = new HashMap<>();
        boolean specifiedModelIds = modelIds != null && modelIds.length > 0;
        String[] removedModelIds = specifiedModelIds ? modelIds : mlModelManager.getAllModelIds();
        if (removedModelIds != null) {
            for (String modelId : removedModelIds) {
                String[] workerNodes = mlModelManager.getWorkerNodes(modelId);
                modelWorkerNodeCounts.put(modelId, workerNodes == null ? 0 : workerNodes.length);
            }
        }

        Map<String, String> modelUndeployStatus = mlModelManager.undeployModel(modelIds);
        mlStats.getStat(MLNodeLevelStat.ML_NODE_EXECUTING_TASK_COUNT).decrement();
        return new MLUndeployModelNodeResponse(clusterService.localNode(), modelUndeployStatus, modelWorkerNodeCounts);
    }
}
