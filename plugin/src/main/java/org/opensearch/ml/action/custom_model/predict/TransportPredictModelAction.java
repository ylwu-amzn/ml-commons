/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.custom_model.predict;

import static org.opensearch.ml.plugin.MachineLearningPlugin.TASK_THREAD_POOL;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionListenerResponseHandler;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.action.support.ThreadedActionListener;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardAction;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardInput;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardRequest;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardRequestType;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardResponse;
import org.opensearch.ml.common.transport.custom_model.predict.MLPredictModelAction;
import org.opensearch.ml.common.transport.custom_model.predict.MLPredictModelRequest;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.engine.algorithms.custom.CustomModelManager;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.task.MLTaskDispatcher;
import org.opensearch.tasks.Task;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

@Log4j2
public class TransportPredictModelAction extends HandledTransportAction<ActionRequest, MLTaskResponse> {
    TransportService transportService;
    CustomModelManager customModelManager;
    MLTaskDispatcher mlTaskDispatcher;
    MLModelManager mlModelManager;
    ClusterService clusterService;
    ThreadPool threadPool;

    @Inject
    public TransportPredictModelAction(
        TransportService transportService,
        ActionFilters actionFilters,
        CustomModelManager customModelManager,
        MLTaskDispatcher mlTaskDispatcher,
        MLModelManager mlModelManager,
        ClusterService clusterService,
        ThreadPool threadPool
    ) {
        super(MLPredictModelAction.NAME, transportService, actionFilters, MLPredictModelRequest::new);
        this.transportService = transportService;
        this.customModelManager = customModelManager;
        this.mlTaskDispatcher = mlTaskDispatcher;
        this.mlModelManager = mlModelManager;
        this.clusterService = clusterService;
        this.threadPool = threadPool;
    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<MLTaskResponse> listener) {
        MLPredictionTaskRequest mlPredictModelRequest = MLPredictionTaskRequest.fromActionRequest(request);
        String modelId = mlPredictModelRequest.getModelId();
        MLInput input = mlPredictModelRequest.getMlInput();
        try {
            ActionListener<String> actionListener = ActionListener.wrap(nodeId -> {
                if (clusterService.localNode().getId().equals(nodeId)) {
                    ModelTensorOutput result = customModelManager.predict(modelId, input);
                    listener.onResponse(MLTaskResponse.builder().output(result).build());
                } else {
                    MLForwardInput mlForwardInput = MLForwardInput
                        .builder()
                        .requestType(MLForwardRequestType.PREDICT_MODEL)
                        .modelId(modelId)
                        .modelInput(input)
                        .build();
                    MLForwardRequest forwardRequest = new MLForwardRequest(mlForwardInput);
                    ActionListener<MLForwardResponse> myListener = ActionListener
                        .wrap(
                            res -> { listener.onResponse(MLTaskResponse.builder().output(res.getMlOutput()).build()); },
                            ex -> { listener.onFailure(ex); }
                        );
                    ThreadedActionListener threadedActionListener = new ThreadedActionListener<>(
                        log,
                        threadPool,
                        TASK_THREAD_POOL,
                        myListener,
                        false
                    );
                    transportService
                        .sendRequest(
                            mlTaskDispatcher.getNode(nodeId),
                            MLForwardAction.NAME,
                            forwardRequest,
                            new ActionListenerResponseHandler<MLForwardResponse>(threadedActionListener, MLForwardResponse::new)
                        );
                }
            }, e -> { listener.onFailure(e); });
            String[] workerNodes = mlModelManager.getWorkerNodes(modelId);
            mlTaskDispatcher.dispatchModel(workerNodes, actionListener);
        } catch (Exception e) {
            log.error("Failed to download custom model " + modelId, e);
            listener.onFailure(e);
        }
    }
}
