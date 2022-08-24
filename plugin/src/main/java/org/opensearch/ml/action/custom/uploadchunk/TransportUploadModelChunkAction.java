/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.custom.uploadchunk;

import lombok.extern.log4j.Log4j2;
import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.MLTaskState;
import org.opensearch.ml.common.MLTaskType;
import org.opensearch.ml.common.dataset.MLInputDataType;
import org.opensearch.ml.common.transport.custom.load.LoadModelResponse;
import org.opensearch.ml.common.transport.custom.uploadchunk.MLUploadChunkInput;
import org.opensearch.ml.common.transport.custom.uploadchunk.MLUploadModelChunkAction;
import org.opensearch.ml.common.transport.custom.uploadchunk.MLUploadModelChunkRequest;
import org.opensearch.ml.engine.algorithms.custom.CustomModelManager;
import org.opensearch.ml.indices.MLIndicesHandler;
import org.opensearch.ml.task.MLTaskDispatcher;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.tasks.Task;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.time.Instant;

@Log4j2
public class TransportUploadModelChunkAction extends HandledTransportAction<ActionRequest, LoadModelResponse> {
    TransportService transportService;
    CustomModelManager customModelManager;
    MLIndicesHandler mlIndicesHandler;
    MLTaskManager mlTaskManager;
    ClusterService clusterService;
    ThreadPool threadPool;
    Client client;
    MLTaskDispatcher mlTaskDispatcher;
    MLModelChunkUploader mlModelUploader;

    @Inject
    public TransportUploadModelChunkAction(
            TransportService transportService,
            ActionFilters actionFilters,
            CustomModelManager customModelManager,
            MLIndicesHandler mlIndicesHandler,
            MLTaskManager mlTaskManager,
            ClusterService clusterService,
            ThreadPool threadPool,
            Client client,
            MLTaskDispatcher mlTaskDispatcher,
            MLModelChunkUploader mlModelUploader
    ) {
        super(MLUploadModelChunkAction.NAME, transportService, actionFilters, MLUploadModelChunkRequest::new);
        this.transportService = transportService;
        this.customModelManager = customModelManager;
        this.mlIndicesHandler = mlIndicesHandler;
        this.mlTaskManager = mlTaskManager;
        this.clusterService = clusterService;
        this.threadPool = threadPool;
        this.client = client;
        this.mlTaskDispatcher = mlTaskDispatcher;
        this.mlModelUploader = mlModelUploader;
    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<LoadModelResponse> listener) {
        MLUploadModelChunkRequest uploadModelRequest = MLUploadModelChunkRequest.fromActionRequest(request);
        MLUploadChunkInput mlUploadChunkInput = uploadModelRequest.getMlUploadInput();

        try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
            MLTask mlTask = MLTask.builder()
                    .async(true)
                    .taskType(MLTaskType.UPLOAD_MODEL)
                    .functionName(FunctionName.CUSTOM)
                    .inputType(MLInputDataType.SEARCH_QUERY)
                    .createTime(Instant.now())
                    .lastUpdateTime(Instant.now())
                    .state(MLTaskState.CREATED)//TODO: mark task as done or failed
                    .workerNode(clusterService.localNode().getId())
                    .build();
            mlModelUploader.uploadModel(mlUploadChunkInput, listener);
        } catch (Exception e) {
            log.error("Failed to upload ML model", e);
            listener.onFailure(e);
        }
    }
}
