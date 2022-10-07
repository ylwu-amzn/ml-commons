/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.custom_model.load;

import static org.opensearch.ml.plugin.MachineLearningPlugin.TASK_THREAD_POOL;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import lombok.extern.log4j.Log4j2;

import org.apache.commons.lang3.exception.ExceptionUtils;
import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.client.Client;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.MLTaskState;
import org.opensearch.ml.common.MLTaskType;
import org.opensearch.ml.common.dataset.MLInputDataType;
import org.opensearch.ml.common.exception.MLResourceNotFoundException;
import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.common.transport.custom_model.load.LoadModelInput;
import org.opensearch.ml.common.transport.custom_model.load.LoadModelNodesRequest;
import org.opensearch.ml.common.transport.custom_model.load.LoadModelNodesResponse;
import org.opensearch.ml.common.transport.custom_model.load.LoadModelResponse;
import org.opensearch.ml.common.transport.custom_model.load.MLLoadModelAction;
import org.opensearch.ml.common.transport.custom_model.load.MLLoadModelOnNodeAction;
import org.opensearch.ml.common.transport.custom_model.load.MLLoadModelRequest;
import org.opensearch.ml.engine.algorithms.custom.CustomModelManager;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.task.MLTaskDispatcher;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.tasks.Task;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import com.google.common.collect.ImmutableMap;

@Log4j2
public class TransportLoadModelAction extends HandledTransportAction<ActionRequest, LoadModelResponse> {
    TransportService transportService;
    CustomModelManager customModelManager;
    MLTaskManager mlTaskManager;
    ClusterService clusterService;
    ThreadPool threadPool;
    Client client;
    NamedXContentRegistry xContentRegistry;
    MLTaskDispatcher mlTaskDispatcher;
    MLModelManager mlModelManager;

    @Inject
    public TransportLoadModelAction(
        TransportService transportService,
        ActionFilters actionFilters,
        CustomModelManager customModelManager,
        MLTaskManager mlTaskManager,
        ClusterService clusterService,
        ThreadPool threadPool,
        Client client,
        NamedXContentRegistry xContentRegistry,
        MLTaskDispatcher mlTaskDispatcher,
        MLModelManager mlModelManager
    ) {
        super(MLLoadModelAction.NAME, transportService, actionFilters, MLLoadModelRequest::new);
        this.transportService = transportService;
        this.customModelManager = customModelManager;
        this.mlTaskManager = mlTaskManager;
        this.clusterService = clusterService;
        this.threadPool = threadPool;
        this.client = client;
        this.xContentRegistry = xContentRegistry;
        this.mlTaskDispatcher = mlTaskDispatcher;
        this.mlModelManager = mlModelManager;
    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<LoadModelResponse> listener) {
        MLLoadModelRequest deployModelRequest = MLLoadModelRequest.fromActionRequest(request);
        String modelId = deployModelRequest.getModelId();
        String[] targetNodeIds = deployModelRequest.getModelNodeIds();
        try {
            DiscoveryNode[] allEligibleNodes = mlTaskDispatcher.getEligibleNodes();
            Map<String, DiscoveryNode> nodeMapping = new HashMap();
            for (DiscoveryNode node : allEligibleNodes) {
                nodeMapping.put(node.getId(), node);
            }

            Set<String> allEligibleNodeIds = Arrays.stream(allEligibleNodes).map(n -> n.getId()).collect(Collectors.toSet());

            List<DiscoveryNode> eligibleNodes = new ArrayList<>();
            List<String> nodeIds = new ArrayList<>();
            if (targetNodeIds != null) {
                for (String nodeId : targetNodeIds) {
                    if (allEligibleNodeIds.contains(nodeId)) {
                        eligibleNodes.add(nodeMapping.get(nodeId));
                        nodeIds.add(nodeId);
                    }
                }
            } else {
                nodeIds.addAll(allEligibleNodeIds);
                for (DiscoveryNode node : allEligibleNodes) {
                    eligibleNodes.add(node);
                }
            }
            if (nodeIds.size() == 0) {
                listener.onFailure(new MLResourceNotFoundException("no eligible node found"));
                return;
            }

            String workerNodes = String.join(",", nodeIds);
            log.warn("Will load model on these nodes: {}", workerNodes);
            String localNodeId = clusterService.localNode().getId();

            String[] includes = new String[] { MLModel.MODEL_NAME_FIELD, MLModel.MODEL_VERSION_FIELD, MLModel.ALGORITHM_FIELD };
            try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                mlModelManager.getModel(modelId, includes, null, ActionListener.wrap(mlModel -> {
                    FunctionName algorithm = mlModel.getAlgorithm();
                    MLTask mlTask = MLTask
                        .builder()
                        .async(true)
                        .modelId(modelId)
                        .taskType(MLTaskType.LOAD_MODEL)
                        .functionName(algorithm)
                        .inputType(MLInputDataType.SEARCH_QUERY)
                        .createTime(Instant.now())
                        .lastUpdateTime(Instant.now())
                        .state(MLTaskState.CREATED)
                        .workerNode(localNodeId)
                        .build();
                    mlTaskManager.createMLTask(mlTask, ActionListener.wrap(response -> {
                        String taskId = response.getId();
                        mlTask.setTaskId(taskId);

                        try {
                            mlTaskManager.add(mlTask, nodeIds);
                            listener.onResponse(new LoadModelResponse(taskId, MLTaskState.CREATED.name()));
                            threadPool.executor(TASK_THREAD_POOL).execute(() -> {
                                LoadModelInput loadModelInput = new LoadModelInput(
                                    modelId,
                                    taskId,
                                    eligibleNodes.size(),
                                    localNodeId,
                                    mlTask
                                );
                                LoadModelNodesRequest loadModelRequest = new LoadModelNodesRequest(
                                    eligibleNodes.toArray(new DiscoveryNode[0]),
                                    loadModelInput
                                );
                                ActionListener<LoadModelNodesResponse> actionListener = ActionListener.wrap(r -> {
                                    if (mlTaskManager.contains(taskId)) {
                                        mlTaskManager.updateMLTask(taskId, ImmutableMap.of(MLTask.STATE_FIELD, MLTaskState.RUNNING), 5000);
                                    }
                                }, e -> {
                                    log.error("Failed to load model " + modelId, e);
                                    mlTaskManager
                                        .updateMLTask(
                                            taskId,
                                            ImmutableMap
                                                .of(
                                                    MLTask.ERROR_FIELD,
                                                    ExceptionUtils.getStackTrace(e),
                                                    MLTask.STATE_FIELD,
                                                    MLTaskState.FAILED
                                                ),
                                            5000
                                        );
                                    mlTaskManager.remove(taskId);
                                });
                                mlModelManager
                                    .updateModel(
                                        modelId,
                                        ImmutableMap.of(MLModel.MODEL_STATE_FIELD, MLModelState.LOADING),
                                        ActionListener
                                            .wrap(
                                                r -> {
                                                    client.execute(MLLoadModelOnNodeAction.INSTANCE, loadModelRequest, actionListener);
                                                },
                                                e -> {
                                                    e.printStackTrace();
                                                    actionListener.onFailure(e);
                                                }
                                            )
                                    );
                            });
                        } catch (Exception ex) {
                            log.error("Failed to load custom model", ex);
                            mlTaskManager.remove(taskId);
                            listener.onFailure(ex);
                        }
                    }, exception -> {
                        log.error("Failed to create upload model task for " + modelId, exception);
                        listener.onFailure(exception);
                    }));
                }, e -> {
                    log.error("Failed to get model " + modelId, e);
                    listener.onFailure(e);
                }));
            } catch (Exception e) {
                log.error("Failed to load " + modelId, e);
                listener.onFailure(e);
            }
        } catch (Exception e) {
            log.error("Failed to download custom model " + modelId, e);
            listener.onFailure(e);
        }
    }

}
