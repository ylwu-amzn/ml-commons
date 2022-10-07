/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.custom_model.forward;

import java.time.Instant;
import java.util.List;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.ml.action.custom_model.upload.MLModelUploader;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.MLTaskState;
import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardAction;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardInput;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardRequest;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardRequestType;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardResponse;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpAction;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpRequest;
import org.opensearch.ml.common.transport.custom_model.upload.MLUploadInput;
import org.opensearch.ml.engine.algorithms.custom.CustomModelManager;
import org.opensearch.ml.indices.MLIndicesHandler;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.task.MLTaskCache;
import org.opensearch.ml.task.MLTaskDispatcher;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.tasks.Task;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import com.google.common.collect.ImmutableMap;

@Log4j2
public class TransportForwardAction extends HandledTransportAction<ActionRequest, MLForwardResponse> {
    TransportService transportService;
    CustomModelManager customModelManager;
    MLTaskManager mlTaskManager;
    ClusterService clusterService;
    ThreadPool threadPool;
    Client client;
    NamedXContentRegistry xContentRegistry;
    MLTaskDispatcher mlTaskDispatcher;
    MLIndicesHandler mlIndicesHandler;
    MLModelUploader mlModelUploader;
    MLModelManager mlModelManager;

    @Inject
    public TransportForwardAction(
        TransportService transportService,
        ActionFilters actionFilters,
        CustomModelManager customModelManager,
        MLTaskManager mlTaskManager,
        ClusterService clusterService,
        ThreadPool threadPool,
        Client client,
        NamedXContentRegistry xContentRegistry,
        MLTaskDispatcher mlTaskDispatcher,
        MLIndicesHandler mlIndicesHandler,
        MLModelUploader mlModelUploader,
        MLModelManager mlModelManager
    ) {
        super(MLForwardAction.NAME, transportService, actionFilters, MLForwardRequest::new);
        this.transportService = transportService;
        this.customModelManager = customModelManager;
        this.mlTaskManager = mlTaskManager;
        this.clusterService = clusterService;
        this.threadPool = threadPool;
        this.client = client;
        this.xContentRegistry = xContentRegistry;
        this.mlTaskDispatcher = mlTaskDispatcher;
        this.mlIndicesHandler = mlIndicesHandler;
        this.mlModelUploader = mlModelUploader;
        this.mlModelManager = mlModelManager;
    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<MLForwardResponse> listener) {
        MLForwardRequest mlForwardRequest = MLForwardRequest.fromActionRequest(request);
        MLForwardInput forwardInput = mlForwardRequest.getForwardInput();
        String modelId = forwardInput.getModelId();
        String taskId = forwardInput.getTaskId();
        MLUploadInput uploadInput = forwardInput.getUploadInput();
        MLTask mlTask = forwardInput.getMlTask();
        String workerNodeId = forwardInput.getWorkerNodeId();
        MLForwardRequestType requestType = forwardInput.getRequestType();

        String error = forwardInput.getError();
        log.info("receive forward request: " + forwardInput.getRequestType());
        try {
            switch (requestType) {
                case LOAD_MODEL_DONE:
                    List<String> workNodes = mlTaskManager.getWorkNodes(taskId);
                    if (workNodes != null) {
                        workNodes.remove(workerNodeId);
                    }

                    if (error != null) {
                        mlTaskManager.addNodeError(taskId, workerNodeId, error);
                    } else {
                        mlModelManager.addModelWorkerNode(modelId, workerNodeId);
                    }

                    if (workNodes == null || workNodes.size() == 0) {
                        MLTaskCache mlTaskCache = mlTaskManager.getMLTaskCache(taskId);
                        MLTaskState taskState = mlTaskCache.hasError() ? MLTaskState.COMPLETED_WITH_ERROR : MLTaskState.COMPLETED;
                        if (mlTaskCache.allNodeFailed()) {
                            taskState = MLTaskState.FAILED;
                        } else {
                            String[] allNodes = mlTaskDispatcher.getAllNodes();
                            String[] workerNodes = mlModelManager.getWorkerNodes(modelId);
                            if (allNodes.length > 1 && workerNodes.length > 0) {
                                MLSyncUpRequest syncUpRequest = new MLSyncUpRequest(modelId, workerNodes, false);
                                client
                                    .execute(
                                        MLSyncUpAction.INSTANCE,
                                        syncUpRequest,
                                        ActionListener
                                            .wrap(r -> { log.debug("Sync up successfully"); }, e -> { log.error("Failed to sync up", e); })
                                    );
                            }
                        }
                        ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();
                        builder.put(MLTask.STATE_FIELD, taskState);
                        if (mlTaskCache.hasError()) {
                            builder.put(MLTask.ERROR_FIELD, mlTaskCache.getErrors().toString());
                        }
                        mlTaskManager.updateMLTask(taskId, builder.build(), 5000);

                        if (!mlTaskCache.allNodeFailed()) {
                            MLModelState modelState = mlTaskCache.hasError() ? MLModelState.PARTIALLY_LOADED : MLModelState.LOADED;
                            mlModelManager
                                .updateModel(
                                    modelId,
                                    ImmutableMap
                                        .of(
                                            MLModel.MODEL_STATE_FIELD,
                                            modelState,
                                            MLModel.LATEST_LOADED_TIME_FIELD,
                                            Instant.now().toEpochMilli()
                                        )
                                );
                        }

                        mlTaskManager.remove(taskId); // TODO: change task state as finished.
                    }
                    listener.onResponse(new MLForwardResponse("ok", null));
                    break;
                case UPLOAD_MODEL:
                    mlModelUploader.newUploadMoadel(uploadInput, mlTask);
                    listener.onResponse(new MLForwardResponse("ok", null));
                    break;
                case PREDICT_MODEL:
                    ModelTensorOutput output = customModelManager.predict(forwardInput.getModelId(), forwardInput.getModelInput());
                    listener.onResponse(new MLForwardResponse("ok", output));
                    break;
                default:
                    throw new IllegalArgumentException("unsupported request type");
            }
        } catch (Exception e) {
            log.error("Failed to execute forward action", e);
            listener.onFailure(e);
        }
    }
}
