/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.upload;

import static org.opensearch.ml.common.CommonValue.ML_MODEL_INDEX;
import static org.opensearch.ml.engine.MLEngine.getUploadModelPath;
import static org.opensearch.ml.engine.ModelHelper.CHUNK_FILES;
import static org.opensearch.ml.engine.ModelHelper.MODEL_FILE_HASH;
import static org.opensearch.ml.engine.ModelHelper.MODEL_SIZE_IN_BYTES;
import static org.opensearch.ml.engine.utils.FileUtils.deleteFileQuietly;
import static org.opensearch.ml.plugin.MachineLearningPlugin.TASK_THREAD_POOL;
import static org.opensearch.ml.settings.MLCommonsSettings.ML_COMMONS_MAX_UPLOAD_TASKS_PER_NODE;

import java.io.File;
import java.time.Instant;
import java.util.Arrays;
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
import org.opensearch.action.support.IndicesOptions;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.index.reindex.DeleteByQueryAction;
import org.opensearch.index.reindex.DeleteByQueryRequest;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.MLTaskState;
import org.opensearch.ml.common.MLTaskType;
import org.opensearch.ml.common.breaker.MLCircuitBreakerService;
import org.opensearch.ml.common.exception.MLLimitExceededException;
import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.common.transport.load.MLLoadModelAction;
import org.opensearch.ml.common.transport.load.MLLoadModelRequest;
import org.opensearch.ml.common.transport.upload.MLUploadInput;
import org.opensearch.ml.engine.ModelHelper;
import org.opensearch.ml.indices.MLIndicesHandler;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.stats.ActionName;
import org.opensearch.ml.stats.MLActionLevelStat;
import org.opensearch.ml.stats.MLNodeLevelStat;
import org.opensearch.ml.stats.MLStats;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.threadpool.ThreadPool;

import com.google.common.collect.ImmutableMap;
import com.google.common.io.Files;

@Log4j2
public class MLModelUploader {

    public static final int TIMEOUT_IN_MILLIS = 5000;
    private final ModelHelper modelHelper;
    private final MLIndicesHandler mlIndicesHandler;
    private final MLTaskManager mlTaskManager;
    private final MLModelManager mlModelManager;
    private final ThreadPool threadPool;
    private final Client client;
    private final MLStats mlStats;
    protected final MLCircuitBreakerService mlCircuitBreakerService;
    private volatile Integer maxUploadTasksPerNode;

    public MLModelUploader(
        ModelHelper modelHelper,
        MLIndicesHandler mlIndicesHandler,
        MLTaskManager mlTaskManager,
        MLModelManager mlModelManager,
        ThreadPool threadPool,
        Client client,
        MLStats mlStats,
        MLCircuitBreakerService mlCircuitBreakerService,
        ClusterService clusterService,
        Settings settings
    ) {
        this.modelHelper = modelHelper;
        this.mlIndicesHandler = mlIndicesHandler;
        this.mlTaskManager = mlTaskManager;
        this.mlModelManager = mlModelManager;
        this.threadPool = threadPool;
        this.client = client;
        this.mlStats = mlStats;
        this.mlCircuitBreakerService = mlCircuitBreakerService;

        maxUploadTasksPerNode = ML_COMMONS_MAX_UPLOAD_TASKS_PER_NODE.get(settings);
        clusterService.getClusterSettings().addSettingsUpdateConsumer(ML_COMMONS_MAX_UPLOAD_TASKS_PER_NODE, it -> maxUploadTasksPerNode = it);
    }

    public void uploadMLModel(MLUploadInput uploadInput, MLTask mlTask) {
//        if (mlTaskManager.getRunningTaskCount(MLTaskType.UPLOAD_MODEL) >= maxUploadTasksPerNode) {
//            String errorMsg = "exceed max upload task limit";
//            mlTaskManager.updateMLTaskDirectly(mlTask.getTaskId(), ImmutableMap.of(MLTask.STATE_FIELD, MLTaskState.FAILED, MLTask.ERROR_FIELD, errorMsg));
//            throw new MLLimitExceededException(errorMsg);
//        }
//        if (mlCircuitBreakerService.isOpen()) {
//            mlStats.getStat(MLNodeLevelStat.ML_NODE_TOTAL_CIRCUIT_BREAKER_TRIGGER_COUNT).increment();
//            String errorMsg = "Circuit breaker is open, please check your memory and disk usage!";
//            mlTaskManager.updateMLTaskDirectly(mlTask.getTaskId(), ImmutableMap.of(MLTask.STATE_FIELD, MLTaskState.FAILED, MLTask.ERROR_FIELD, errorMsg));
//            throw new MLLimitExceededException(errorMsg);
//        }
        mlStats.getStat(MLNodeLevelStat.ML_NODE_TOTAL_REQUEST_COUNT).increment();
        String errorMsg = checkResourceLimit();
        if (errorMsg != null) {
            mlTaskManager.updateMLTaskDirectly(mlTask.getTaskId(), ImmutableMap.of(MLTask.STATE_FIELD, MLTaskState.FAILED, MLTask.ERROR_FIELD, errorMsg));
            throw new MLLimitExceededException(errorMsg);
        }
        mlTaskManager.add(mlTask);
        mlStats.getStat(MLNodeLevelStat.ML_NODE_EXECUTING_TASK_COUNT).increment();
        mlStats
            .createCounterStatIfAbsent(mlTask.getFunctionName(), ActionName.UPLOAD, MLActionLevelStat.ML_ACTION_REQUEST_COUNT)
            .increment();
        try {
            if (uploadInput.getUrl() != null) {
                uploadModel(uploadInput, mlTask);
            } else {
                uploadPrebuiltModel(uploadInput, mlTask);
            }
        } catch (Exception e) {
            mlStats
                .createCounterStatIfAbsent(mlTask.getFunctionName(), ActionName.UPLOAD, MLActionLevelStat.ML_ACTION_FAILURE_COUNT)
                .increment();
        } finally {
            mlStats.getStat(MLNodeLevelStat.ML_NODE_EXECUTING_TASK_COUNT).increment();
        }
    }

    private String checkResourceLimit() {
        if (mlTaskManager.getRunningTaskCount(MLTaskType.UPLOAD_MODEL) >= maxUploadTasksPerNode) {
            return "exceed max upload task limit";
        }
        if (mlCircuitBreakerService.isOpen()) {
            mlStats.getStat(MLNodeLevelStat.ML_NODE_TOTAL_CIRCUIT_BREAKER_TRIGGER_COUNT).increment();
            return "Circuit breaker is open, please check your memory and disk usage!";
        }
        return null;
    }

    public void uploadPrebuiltModel(MLUploadInput uploadInput, MLTask mlTask) {
        String modelName = uploadInput.getModelName();
        String taskId = mlTask.getTaskId();
        Integer version = uploadInput.getVersion();
        boolean loadModel = uploadInput.isLoadModel();
//        mlTaskManager.add(mlTask);
        modelHelper.downloadPrebuiltModelConfig(taskId, uploadInput, ActionListener.wrap(response -> {
            mlTaskManager.remove(taskId);
            uploadModel(response, mlTask);
        }, e -> {
            log.error("Failed to upload pre built model", e);
            handleException(taskId, e);
        }));
    }

    public void uploadModel(MLUploadInput mlUploadInput, MLTask mlTask) {
        Semaphore semaphore = new Semaphore(1);
        String taskId = mlTask.getTaskId();
//        mlTaskManager.add(mlTask);

        AtomicInteger uploaded = new AtomicInteger(0);
        threadPool.executor(TASK_THREAD_POOL).execute(() -> {
            String modelName = mlUploadInput.getModelName();
            Integer version = mlUploadInput.getVersion();

            try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                mlIndicesHandler.initModelIndexIfAbsent(ActionListener.wrap(res -> {
                    MLModel mlModelMeta = MLModel
                        .builder()
                        .name(modelName)
                        .algorithm(mlTask.getFunctionName())
                        .version(version)
                        .modelFormat(mlUploadInput.getModelFormat())
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
                        mlTask.setModelId(modelId);
                        log.info("create new model meta doc {} for upload task {}", modelId, taskId);
                        modelHelper.downloadAndSplit(modelId, modelName, version, mlUploadInput.getUrl(), ActionListener.wrap(result -> {
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
                                    .algorithm(mlTask.getFunctionName())
                                    .version(version)
                                    .modelFormat(mlUploadInput.getModelFormat())
                                    .chunkNumber(chunkNum)
                                    .totalChunks(chunkFiles.size())
                                    .content(Base64.getEncoder().encodeToString(bytes))
                                    .createdTime(Instant.now())
                                    .build();
                                IndexRequest indexRequest = new IndexRequest(ML_MODEL_INDEX);
                                indexRequest.id(mlModelManager.getModelChunkId(modelId, chunkNum));
                                indexRequest
                                    .source(
                                        mlModel.toXContent(XContentBuilder.builder(XContentType.JSON.xContent()), ToXContent.EMPTY_PARAMS)
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
                                                        MLModel.LAST_UPLOADED_TIME_FIELD,
                                                        Instant.now().toEpochMilli(),
                                                        MLModel.TOTAL_CHUNKS_FIELD,
                                                        chunkFiles.size(),
                                                        MLModel.MODEL_CONTENT_HASH_VALUE_FIELD,
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
                                                    if (mlUploadInput.isLoadModel()) {
                                                        String[] modelNodeIds = mlUploadInput.getModelNodeIds();
                                                        log
                                                            .debug(
                                                                "uploading model done, start loading model {} on nodes: {}",
                                                                modelId,
                                                                Arrays.toString(modelNodeIds)
                                                            );
                                                        MLLoadModelRequest mlLoadModelRequest = new MLLoadModelRequest(
                                                            modelId,
                                                            modelNodeIds,
                                                            false,
                                                            true
                                                        );
                                                        client
                                                            .execute(
                                                                MLLoadModelAction.INSTANCE,
                                                                mlLoadModelRequest,
                                                                ActionListener
                                                                    .wrap(
                                                                        response -> { log.info(response); },
                                                                        exc -> { exc.printStackTrace(); }
                                                                    )
                                                            );
                                                    }
                                                }, e -> {
                                                    log.error("Failed to index model chunk", e);
                                                    handleException(taskId, e);
                                                    deleteModel(modelId);
                                                })
                                            );
                                    } else {
                                        file.delete();
                                    }
                                    semaphore.release();
                                }, e -> {
                                    log.error("Failed to index model chunk", e);
                                    handleException(taskId, e);
                                    file.delete();
                                    // remove model doc as failed to upload model
                                    deleteModel(modelId);
                                    semaphore.release();
                                    deleteFileQuietly(getUploadModelPath(modelId));
                                }));
                            }
                        }, e -> {
                            log.error("Failed to index chunk file", e);
                            deleteFileQuietly(getUploadModelPath(modelId));
                            deleteModel(modelId);
                            handleException(taskId, e);
                        }));
                    }, e -> {
                        log.error("Failed to index model meta doc", e);
                        handleException(taskId, e);
                    }));
                }, e -> {
                    log.error("Failed to init model index", e);
                    handleException(taskId, e);
                }));
            } catch (Exception e) {
                log.error("Failed to upload model", e);
                handleException(taskId, e);
            }
        });
    }

    private void deleteModel(String modelId) {
        DeleteRequest deleteRequest = new DeleteRequest();
        deleteRequest.index(ML_MODEL_INDEX).id(modelId).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
        client.delete(deleteRequest);
        DeleteByQueryRequest deleteChunksRequest = new DeleteByQueryRequest(ML_MODEL_INDEX)
            .setQuery(new TermQueryBuilder(MLModel.MODEL_ID_FIELD, modelId))
            .setIndicesOptions(IndicesOptions.LENIENT_EXPAND_OPEN)
            .setAbortOnVersionConflict(false);
        client.execute(DeleteByQueryAction.INSTANCE, deleteChunksRequest);
    }

    private void handleException(String taskId, Exception e) {
        mlTaskManager
            .updateMLTask(
                taskId,
                ImmutableMap.of(MLTask.ERROR_FIELD, ExceptionUtils.getStackTrace(e), MLTask.STATE_FIELD, MLTaskState.FAILED),
                ActionListener.runAfter(ActionListener.wrap(r -> {
                    log.debug("updated task successfully {}", taskId);
                }, ex->{
                    log.error("failed to update ML task " + taskId, ex);
                }), ()-> mlTaskManager.remove(taskId)),
                TIMEOUT_IN_MILLIS
            );
    }
}
