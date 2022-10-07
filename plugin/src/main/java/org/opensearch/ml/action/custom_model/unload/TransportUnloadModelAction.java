/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.custom_model.unload;

import static org.opensearch.ml.common.CommonValue.DELETED;
import static org.opensearch.ml.common.CommonValue.NOT_FOUND;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.action.FailedNodeException;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.nodes.TransportNodesAction;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpAction;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpRequest;
import org.opensearch.ml.common.transport.custom_model.unload.MLUnloadModelAction;
import org.opensearch.ml.common.transport.custom_model.unload.UnloadModelInput;
import org.opensearch.ml.common.transport.custom_model.unload.UnloadModelNodeRequest;
import org.opensearch.ml.common.transport.custom_model.unload.UnloadModelNodeResponse;
import org.opensearch.ml.common.transport.custom_model.unload.UnloadModelNodesRequest;
import org.opensearch.ml.common.transport.custom_model.unload.UnloadModelNodesResponse;
import org.opensearch.ml.engine.algorithms.custom.CustomModelManager;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

@Log4j2
public class TransportUnloadModelAction extends
    TransportNodesAction<UnloadModelNodesRequest, UnloadModelNodesResponse, UnloadModelNodeRequest, UnloadModelNodeResponse> {
    private final CustomModelManager customModelManager;
    private final MLModelManager mlModelManager;
    private final ClusterService clusterService;
    private final Client client;

    @Inject
    public TransportUnloadModelAction(
        TransportService transportService,
        ActionFilters actionFilters,
        CustomModelManager customModelManager,
        MLModelManager mlModelManager,
        ClusterService clusterService,
        ThreadPool threadPool,
        Client client
    ) {
        super(
            MLUnloadModelAction.NAME,
            threadPool,
            clusterService,
            transportService,
            actionFilters,
            UnloadModelNodesRequest::new,
            UnloadModelNodeRequest::new,
            ThreadPool.Names.MANAGEMENT,
            UnloadModelNodeResponse.class
        );
        this.customModelManager = customModelManager;
        this.mlModelManager = mlModelManager;
        this.clusterService = clusterService;
        this.client = client;
    }

    @Override
    protected UnloadModelNodesResponse newResponse(
        UnloadModelNodesRequest nodesRequest,
        List<UnloadModelNodeResponse> responses,
        List<FailedNodeException> failures
    ) {
        if (responses != null) {
            Map<String, List<String>> removedNodeMap = new HashMap<>();
            responses.stream().forEach(r -> {
                Map<String, String> modelUnloadStatus = r.getModelUnloadStatus();
                for (Map.Entry<String, String> entry : modelUnloadStatus.entrySet()) {
                    String status = entry.getValue();
                    if (DELETED.equals(status) || NOT_FOUND.equals(status)) {
                        String modelId = entry.getKey();
                        if (!removedNodeMap.containsKey(modelId)) {
                            removedNodeMap.put(modelId, new ArrayList<>());
                        }
                        removedNodeMap.get(modelId).add(r.getNode().getId());
                    }
                }
            });
            for (Map.Entry<String, List<String>> entry : removedNodeMap.entrySet()) {
                String modelId = entry.getKey();
                List<String> removedNodes = entry.getValue();
                MLSyncUpRequest syncUpRequest = new MLSyncUpRequest(modelId, null, removedNodes.toArray(new String[0]), false, true);
                client
                    .execute(
                        MLSyncUpAction.INSTANCE,
                        syncUpRequest,
                        ActionListener
                            .wrap(r -> { log.info("sync up removed nodes"); }, e -> { log.error("failed to sync up removed node", e); })
                    );
            }
        }
        return new UnloadModelNodesResponse(clusterService.getClusterName(), responses, failures);
    }

    @Override
    protected UnloadModelNodeRequest newNodeRequest(UnloadModelNodesRequest request) {
        return new UnloadModelNodeRequest(request);
    }

    @Override
    protected UnloadModelNodeResponse newNodeResponse(StreamInput in) throws IOException {
        return new UnloadModelNodeResponse(in);
    }

    @Override
    protected UnloadModelNodeResponse nodeOperation(UnloadModelNodeRequest request) {
        return createUnloadModelNodeResponse(request.getUnloadModelNodesRequest());
    }

    private UnloadModelNodeResponse createUnloadModelNodeResponse(UnloadModelNodesRequest unloadModelNodesRequest) {
        UnloadModelInput unloadModelInput = unloadModelNodesRequest.getUnloadModelInput();
        String[] nodeIds = unloadModelInput.getNodeIds();
        Map<String, String> modelUnloadStatus = new HashMap<>();
        if (nodeIds != null && nodeIds.length > 0) {
            Set<String> targetNodeIds = new HashSet<>(Arrays.asList(nodeIds));
            String localNodeId = clusterService.localNode().getId();
            if (!targetNodeIds.contains(localNodeId)) {
                return new UnloadModelNodeResponse(clusterService.localNode(), modelUnloadStatus);
            }
        }

        Map<String, String> modelStatus = mlModelManager.unloadModel(unloadModelInput);
        modelUnloadStatus.putAll(modelStatus);
        Map<String, String> status = customModelManager.unloadModel(unloadModelInput);
        status.entrySet().forEach(entry -> {
            if (!NOT_FOUND.equals(entry.getValue())) {
                modelUnloadStatus.put(entry.getKey(), entry.getValue());
            }
        });
        return new UnloadModelNodeResponse(clusterService.localNode(), modelUnloadStatus);
    }
}
