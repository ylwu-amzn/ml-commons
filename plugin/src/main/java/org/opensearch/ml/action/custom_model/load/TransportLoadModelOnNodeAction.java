/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.custom_model.load;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import lombok.extern.log4j.Log4j2;

import org.apache.commons.lang3.exception.ExceptionUtils;
import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionListenerResponseHandler;
import org.opensearch.action.FailedNodeException;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.nodes.TransportNodesAction;
import org.opensearch.client.Client;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.node.DiscoveryNodes;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.MLTaskState;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardAction;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardInput;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardRequest;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardRequestType;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardResponse;
import org.opensearch.ml.common.transport.custom_model.load.LoadModelInput;
import org.opensearch.ml.common.transport.custom_model.load.LoadModelNodeRequest;
import org.opensearch.ml.common.transport.custom_model.load.LoadModelNodeResponse;
import org.opensearch.ml.common.transport.custom_model.load.LoadModelNodesRequest;
import org.opensearch.ml.common.transport.custom_model.load.LoadModelNodesResponse;
import org.opensearch.ml.common.transport.custom_model.load.MLLoadModelOnNodeAction;
import org.opensearch.ml.engine.algorithms.custom.CustomModelManager;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import com.google.common.collect.ImmutableMap;

@Log4j2
public class TransportLoadModelOnNodeAction extends
    TransportNodesAction<LoadModelNodesRequest, LoadModelNodesResponse, LoadModelNodeRequest, LoadModelNodeResponse> {
    TransportService transportService;
    CustomModelManager customModelManager;
    MLTaskManager mlTaskManager;
    MLModelManager mlModelManager;
    ClusterService clusterService;
    ThreadPool threadPool;
    Client client;
    NamedXContentRegistry xContentRegistry;

    @Inject
    public TransportLoadModelOnNodeAction(
        TransportService transportService,
        ActionFilters actionFilters,
        CustomModelManager customModelManager,
        MLTaskManager mlTaskManager,
        MLModelManager mlModelManager,
        ClusterService clusterService,
        ThreadPool threadPool,
        Client client,
        NamedXContentRegistry xContentRegistry
    ) {
        super(
            MLLoadModelOnNodeAction.NAME,
            threadPool,
            clusterService,
            transportService,
            actionFilters,
            LoadModelNodesRequest::new,
            LoadModelNodeRequest::new,
            ThreadPool.Names.MANAGEMENT,
            LoadModelNodeResponse.class
        );
        this.transportService = transportService;
        this.customModelManager = customModelManager;
        this.mlTaskManager = mlTaskManager;
        this.mlModelManager = mlModelManager;
        this.clusterService = clusterService;
        this.threadPool = threadPool;
        this.client = client;
        this.xContentRegistry = xContentRegistry;
    }

    @Override
    protected LoadModelNodesResponse newResponse(
        LoadModelNodesRequest nodesRequest,
        List<LoadModelNodeResponse> responses,
        List<FailedNodeException> failures
    ) {
        return new LoadModelNodesResponse(clusterService.getClusterName(), responses, failures);
    }

    @Override
    protected LoadModelNodeRequest newNodeRequest(LoadModelNodesRequest request) {
        return new LoadModelNodeRequest(request);
    }

    @Override
    protected LoadModelNodeResponse newNodeResponse(StreamInput in) throws IOException {
        return new LoadModelNodeResponse(in);
    }

    @Override
    protected LoadModelNodeResponse nodeOperation(LoadModelNodeRequest request) {
        return createLoadModelNodeResponse(request.getLoadModelNodesRequest());
    }

    private LoadModelNodeResponse createLoadModelNodeResponse(LoadModelNodesRequest loadModelNodesRequest) {
        LoadModelInput loadModelInput = loadModelNodesRequest.getLoadModelInput();
        String modelId = loadModelInput.getModelId();
        String taskId = loadModelInput.getTaskId();
        Integer nodeCount = loadModelInput.getNodeCount();
        String coordinatingNodeId = loadModelInput.getCoordinatingNodeId();
        MLTask mlTask = loadModelInput.getMlTask();

        Map<String, String> modelLoadStatus = new HashMap<>();
        modelLoadStatus.put(modelId, "received");

        String localNodeId = clusterService.localNode().getId();

        ActionListener<MLForwardResponse> taskDoneListener = ActionListener
            .wrap(res -> { log.info("load model done " + res); }, ex -> { log.error(ex); });

        loadModel(modelId, localNodeId, coordinatingNodeId, mlTask, ActionListener.wrap(r -> {
            if (!coordinatingNodeId.equals(localNodeId)) {
                mlTaskManager.remove(taskId);
            }
            MLForwardInput mlForwardInput = MLForwardInput
                .builder()
                .requestType(MLForwardRequestType.LOAD_MODEL_DONE)
                .taskId(taskId)
                .modelId(modelId)
                .workerNodeId(clusterService.localNode().getId())
                .build();
            MLForwardRequest loadModelDoneMessage = new MLForwardRequest(mlForwardInput);

            transportService
                .sendRequest(
                    getNodeById(coordinatingNodeId),
                    MLForwardAction.NAME,
                    loadModelDoneMessage,
                    new ActionListenerResponseHandler<MLForwardResponse>(taskDoneListener, MLForwardResponse::new)
                );
        }, e -> {
            log.error("Failed to load model " + modelId, e);
            mlTaskManager
                .updateMLTask(
                    taskId,
                    ImmutableMap.of(MLTask.ERROR_FIELD, ExceptionUtils.getStackTrace(e), MLTask.STATE_FIELD, MLTaskState.FAILED),
                    5000
                );

            if (!coordinatingNodeId.equals(localNodeId)) {
                // remove task cache on worker node
                mlTaskManager.remove(taskId);
            }
            MLForwardInput mlForwardInput = MLForwardInput
                .builder()
                .requestType(MLForwardRequestType.LOAD_MODEL_DONE)
                .taskId(taskId)
                .modelId(modelId)
                .workerNodeId(clusterService.localNode().getId())
                .error(ExceptionUtils.getStackTrace(e))
                .build();
            MLForwardRequest loadModelDoneMessage = new MLForwardRequest(mlForwardInput);

            transportService
                .sendRequest(
                    getNodeById(coordinatingNodeId),
                    MLForwardAction.NAME,
                    loadModelDoneMessage,
                    new ActionListenerResponseHandler<MLForwardResponse>(taskDoneListener, MLForwardResponse::new)
                );
        }));

        return new LoadModelNodeResponse(clusterService.localNode(), modelLoadStatus);
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

    private void loadModel(String modelId, String localNodeId, String coordinatingNodeId, MLTask mlTask, ActionListener<String> listener) {
        try {
            if (!coordinatingNodeId.equals(localNodeId)) {
                mlTaskManager.add(mlTask);
            }
            mlModelManager.loadModel1(modelId, listener);
        } catch (Exception e) {
            log.error("Failed to load custom model " + modelId, e);
            listener.onFailure(e);
        }
    }
}
