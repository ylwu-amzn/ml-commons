/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.models.upload_chunk;

import static org.opensearch.ml.common.CommonValue.ML_MODEL_INDEX;
import static org.opensearch.ml.plugin.MachineLearningPlugin.TASK_THREAD_POOL;

import java.time.Instant;
import java.util.UUID;
import java.util.concurrent.Semaphore;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.client.Client;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.common.model.MLModelTaskType;
import org.opensearch.ml.common.transport.model.upload_chunk.MLUploadModelMetaInput;
import org.opensearch.ml.indices.MLIndicesHandler;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.threadpool.ThreadPool;

@Log4j2
public class MLModelMetaUploader {

    public static final int TIMEOUT_IN_MILLIS = 5000;
    private final MLIndicesHandler mlIndicesHandler;
    private final MLTaskManager mlTaskManager;
    private final ThreadPool threadPool;
    private final Client client;

    public MLModelMetaUploader(MLIndicesHandler mlIndicesHandler, MLTaskManager mlTaskManager, ThreadPool threadPool, Client client) {
        this.mlIndicesHandler = mlIndicesHandler;
        this.mlTaskManager = mlTaskManager;
        this.threadPool = threadPool;
        this.client = client;
    }

    public void uploadModelMeta(MLUploadModelMetaInput mlUploadModelMetaInput, ActionListener<String> listener) {
        Semaphore semaphore = new Semaphore(1);
        threadPool.executor(TASK_THREAD_POOL).execute(() -> {
            try {
                String modelName = mlUploadModelMetaInput.getName();
                Integer version = mlUploadModelMetaInput.getVersion();
                try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                    mlIndicesHandler.initModelIndexIfAbsent(ActionListener.wrap(res -> {
                        semaphore.acquire();
                        MLModel mlModelMeta = MLModel
                            .builder()
                            .name(modelName)
                            .algorithm(FunctionName.TEXT_EMBEDDING)
                            .version(version)
                            .modelFormat(mlUploadModelMetaInput.getModelFormat())
                            .modelTaskType(MLModelTaskType.TEXT_EMBEDDING)
                            .modelState(MLModelState.UPLOADING)
                            .modelConfig(mlUploadModelMetaInput.getModelConfig())
                            .totalChunks(mlUploadModelMetaInput.getTotalChunks())
                            .modelContentHash(mlUploadModelMetaInput.getModelContentHash())
                            .modelContentSizeInBytes(mlUploadModelMetaInput.getModelContentSizeInBytes())
                            .createdTime(Instant.now())
                            .build();
                        IndexRequest indexRequest = new IndexRequest(ML_MODEL_INDEX);
                        indexRequest
                            .source(mlModelMeta.toXContent(XContentBuilder.builder(XContentType.JSON.xContent()), ToXContent.EMPTY_PARAMS));
                        indexRequest.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

                        client.index(indexRequest, ActionListener.wrap(r -> {
                            log.debug("Index model successfully {}", modelName);
                            semaphore.release();
                            listener.onResponse(r.getId());
                        }, e -> {
                            log.error("Failed to index model", e);
                            semaphore.release();
                            listener.onFailure(e);
                        }));
                    }, ex -> {
                        log.error("Failed to init model index", ex);
                        listener.onFailure(ex);
                    }));
                } catch (Exception e) {
                    log.error("Failed to create model meta doc", e);
                    listener.onFailure(e);
                }
            } catch (final Exception e) {
                log.error("Failed to init model index", e);
                listener.onFailure(e);
            }

        });
    }

    public static String customModelId(final String name, final int version) {
        return new StringBuilder().append(name).append("_").append(version).append("_").append(UUID.randomUUID().toString()).toString();
    }
}
