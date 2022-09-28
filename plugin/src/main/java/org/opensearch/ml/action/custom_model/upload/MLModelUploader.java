/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.custom_model.upload;

import static org.opensearch.ml.common.CommonValue.ML_MODEL_INDEX;
import static org.opensearch.ml.engine.MLEngine.getUploadModelPath;
import static org.opensearch.ml.engine.algorithms.custom.CustomModelManager.CHUNK_FILES;
import static org.opensearch.ml.engine.algorithms.custom.CustomModelManager.MODEL_FILE_HASH;
import static org.opensearch.ml.engine.algorithms.custom.CustomModelManager.MODEL_SIZE_IN_BYTES;
import static org.opensearch.ml.engine.utils.MLFileUtils.deleteFileQuietly;
import static org.opensearch.ml.plugin.MachineLearningPlugin.TASK_THREAD_POOL;

import java.io.File;
import java.time.Instant;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import lombok.extern.log4j.Log4j2;

import org.apache.commons.lang3.exception.ExceptionUtils;
import org.opensearch.action.ActionListener;
import org.opensearch.action.delete.DeleteRequest;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.client.Client;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.MLTaskState;
import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.common.model.MLModelTaskType;
import org.opensearch.ml.common.transport.custom_model.upload.MLUploadInput;
import org.opensearch.ml.engine.algorithms.custom.CustomModelManager;
import org.opensearch.ml.indices.MLIndicesHandler;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.threadpool.ThreadPool;

import com.google.common.collect.ImmutableMap;
import com.google.common.io.Files;

@Log4j2
public class MLModelUploader {

    public static final int TIMEOUT_IN_MILLIS = 5000;
    private final CustomModelManager customModelManager;
    private final MLIndicesHandler mlIndicesHandler;
    private final MLTaskManager mlTaskManager;
    private final MLModelManager mlModelManager;
    private final ThreadPool threadPool;
    private final Client client;

    public MLModelUploader(
        CustomModelManager customModelManager,
        MLIndicesHandler mlIndicesHandler,
        MLTaskManager mlTaskManager,
        MLModelManager mlModelManager,
        ThreadPool threadPool,
        Client client
    ) {
        this.customModelManager = customModelManager;
        this.mlIndicesHandler = mlIndicesHandler;
        this.mlTaskManager = mlTaskManager;
        this.mlModelManager = mlModelManager;
        this.threadPool = threadPool;
        this.client = client;
    }

    public void uploadModel(MLUploadInput mlUploadInput, MLTask mlTask) {
        Semaphore semaphore = new Semaphore(1);
        String taskId = mlTask.getTaskId();
        mlTaskManager.add(mlTask);

        AtomicInteger uploaded = new AtomicInteger(0);
        threadPool.executor(TASK_THREAD_POOL).execute(() -> {
            String modelName = mlUploadInput.getName();
            Integer version = mlUploadInput.getVersion();

            try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                mlIndicesHandler.initModelIndexIfAbsent(ActionListener.wrap(res -> {
                    MLModel mlModelMeta = MLModel
                        .builder()
                        .name(modelName)
                        .algorithm(FunctionName.CUSTOM)
                        .version(version)
                        .modelFormat(mlUploadInput.getModelFormat())
                        .modelTaskType(MLModelTaskType.TEXT_EMBEDDING)
                        .modelState(MLModelState.UPLOADING)
                        .modelConfig(mlUploadInput.getModelConfig())
                        .createdTime(Instant.now())
                        .build();
                    IndexRequest indexModelMetaRequest = new IndexRequest(ML_MODEL_INDEX);
                    indexModelMetaRequest
                        .source(mlModelMeta.toXContent(XContentBuilder.builder(XContentType.JSON.xContent()), ToXContent.EMPTY_PARAMS));
                    indexModelMetaRequest.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                    client.index(indexModelMetaRequest, ActionListener.wrap(modelMetaRes -> {
                        String modelId = modelMetaRes.getId();
                        customModelManager
                            .downloadAndSplit(modelId, modelName, version, mlUploadInput.getUrl(), ActionListener.wrap(result -> {
                                Long modelSizeInBytes = (Long) result.get(MODEL_SIZE_IN_BYTES);
                                List<String> chunkFiles = (List<String>) result.get(CHUNK_FILES);
                                String hashValue = (String) result.get(MODEL_FILE_HASH);
                                for (String name : chunkFiles) {
                                    semaphore.tryAcquire(10, TimeUnit.SECONDS);
                                    File file = new File(name);
                                    byte[] bytes = Files.toByteArray(file);
                                    int chunkNum = Integer.parseInt(file.getName());
                                    MLModel mlModel = MLModel
                                        .builder()
                                        .modelId(modelId)
                                        .name(modelName)
                                        .algorithm(FunctionName.CUSTOM)
                                        .version(version)
                                        .modelFormat(mlUploadInput.getModelFormat())
                                        .modelTaskType(MLModelTaskType.TEXT_EMBEDDING) // TODO: get this from mlUploadInput
                                        .chunkNumber(chunkNum)
                                        .totalChunks(chunkFiles.size())
                                        .content(Base64.getEncoder().encodeToString(bytes))
                                        .createdTime(Instant.now())
                                        .build();
                                    IndexRequest indexRequest = new IndexRequest(ML_MODEL_INDEX);
                                    indexRequest.id(mlModelManager.getModelChunkId(modelId, chunkNum));
                                    indexRequest
                                        .source(
                                            mlModel
                                                .toXContent(XContentBuilder.builder(XContentType.JSON.xContent()), ToXContent.EMPTY_PARAMS)
                                        );
                                    indexRequest.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                                    client.index(indexRequest, ActionListener.wrap(r -> {
                                        uploaded.getAndIncrement();
                                        if (uploaded.get() == chunkFiles.size()) {
                                            deleteFileQuietly(getUploadModelPath(modelId));
                                            mlModelManager
                                                .updateModel(
                                                    modelId,
                                                    ImmutableMap
                                                        .of(
                                                            MLModel.MODEL_STATE_FIELD,
                                                            MLModelState.UPLOADED,
                                                            MLModel.LATEST_UPLOADED_TIME_FIELD,
                                                            Instant.now().toEpochMilli(),
                                                            MLModel.TOTAL_CHUNKS_FIELD,
                                                            chunkFiles.size(),
                                                            MLModel.MODEL_CONTENT_HASH_FIELD,
                                                            hashValue,
                                                            MLModel.MODEL_CONTENT_SIZE_IN_BYTES_FIELD,
                                                            modelSizeInBytes
                                                        ),
                                                    ActionListener.wrap(updateResponse -> {
                                                        mlTaskManager
                                                            .updateMLTask(
                                                                taskId,
                                                                ImmutableMap
                                                                    .of(
                                                                        MLTask.STATE_FIELD,
                                                                        MLTaskState.COMPLETED,
                                                                        MLTask.MODEL_ID_FIELD,
                                                                        modelId
                                                                    ),
                                                                TIMEOUT_IN_MILLIS
                                                            );
                                                        mlTaskManager.remove(taskId);
                                                    }, exception -> {
                                                        log.error("Failed to index model chunk", exception);
                                                        mlTaskManager
                                                            .updateMLTask(
                                                                taskId,
                                                                ImmutableMap
                                                                    .of(
                                                                        MLTask.ERROR_FIELD,
                                                                        ExceptionUtils.getStackTrace(exception),
                                                                        MLTask.STATE_FIELD,
                                                                        MLTaskState.FAILED
                                                                    ),
                                                                TIMEOUT_IN_MILLIS
                                                            );
                                                        mlTaskManager.remove(taskId);
                                                        DeleteRequest deleteRequest = new DeleteRequest();
                                                        deleteRequest
                                                            .index(ML_MODEL_INDEX)
                                                            .id(modelId)
                                                            .setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                                                        client.delete(deleteRequest);
                                                        // TODO: delete chunks
                                                    })
                                                );
                                        } else {
                                            file.delete();
                                        }
                                        semaphore.release();
                                    }, e -> {
                                        log.error("Failed to index model chunk", e);
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
                                                TIMEOUT_IN_MILLIS
                                            );
                                        mlTaskManager.remove(taskId);
                                        file.delete();

                                        // remove model doc as failed to upload model
                                        DeleteRequest deleteRequest = new DeleteRequest();
                                        deleteRequest
                                            .index(ML_MODEL_INDEX)
                                            .id(modelId)
                                            .setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                                        client.delete(deleteRequest);
                                        // TODO: also delete chunks
                                        semaphore.release();
                                        deleteFileQuietly(getUploadModelPath(modelId));
                                    }));
                                }
                            }, e -> {
                                log.error("Failed to index chunk file", e);
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
                                        TIMEOUT_IN_MILLIS
                                    );
                                mlTaskManager.remove(taskId);
                                deleteFileQuietly(getUploadModelPath(modelId));
                            }));
                    }, e -> {
                        log.error("Failed to index model meta doc", e);
                        mlTaskManager
                            .updateMLTask(
                                taskId,
                                ImmutableMap
                                    .of(MLTask.ERROR_FIELD, ExceptionUtils.getStackTrace(e), MLTask.STATE_FIELD, MLTaskState.FAILED),
                                TIMEOUT_IN_MILLIS
                            );
                        mlTaskManager.remove(taskId);
                    }));
                }, ex -> {
                    log.error("Failed to init model index", ex);
                    mlTaskManager
                        .updateMLTask(
                            taskId,
                            ImmutableMap.of(MLTask.ERROR_FIELD, ExceptionUtils.getStackTrace(ex), MLTask.STATE_FIELD, MLTaskState.FAILED),
                            TIMEOUT_IN_MILLIS
                        );
                    mlTaskManager.remove(taskId);
                }));
            } catch (Exception e) {
                log.error("Failed to upload model", e);
                mlTaskManager
                    .updateMLTask(
                        taskId,
                        ImmutableMap.of(MLTask.ERROR_FIELD, ExceptionUtils.getStackTrace(e), MLTask.STATE_FIELD, MLTaskState.FAILED),
                        TIMEOUT_IN_MILLIS
                    );
                mlTaskManager.remove(taskId);
            }
        });
    }
}
