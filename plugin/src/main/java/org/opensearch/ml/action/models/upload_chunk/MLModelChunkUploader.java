/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.ml.action.models.upload_chunk;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.common.CommonValue.ML_MODEL_INDEX;
import static org.opensearch.ml.utils.MLNodeUtils.createXContentParserFromRegistry;

import java.util.Base64;
import java.util.concurrent.Semaphore;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.client.Client;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.index.IndexNotFoundException;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.exception.MLResourceNotFoundException;
import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.common.model.MLModelTaskType;
import org.opensearch.ml.common.transport.model.upload_chunk.MLUploadChunkInput;
import org.opensearch.ml.common.transport.model.upload_chunk.MLUploadModelChunkResponse;
import org.opensearch.ml.engine.ModelHelper;
import org.opensearch.ml.indices.MLIndicesHandler;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.threadpool.ThreadPool;

@Log4j2
public class MLModelChunkUploader {

    public static final int TIMEOUT_IN_MILLIS = 5000;
    public static final int MAX_ACCEPTED_CHUNK_SIZE_STRING_LEN = 100000000; /* 100MB */
    private final ModelHelper modelHelper;
    private final MLIndicesHandler mlIndicesHandler;
    private final MLTaskManager mlTaskManager;
    private final ThreadPool threadPool;
    private final Client client;
    private final NamedXContentRegistry xContentRegistry;

    public MLModelChunkUploader(
        ModelHelper modelHelper,
        MLIndicesHandler mlIndicesHandler,
        MLTaskManager mlTaskManager,
        ThreadPool threadPool,
        Client client,
        final NamedXContentRegistry xContentRegistry
    ) {
        this.modelHelper = modelHelper;
        this.mlIndicesHandler = mlIndicesHandler;
        this.mlTaskManager = mlTaskManager;
        this.threadPool = threadPool;
        this.client = client;
        this.xContentRegistry = xContentRegistry;
    }

    public void uploadModel(MLUploadChunkInput mlUploadInput, ActionListener<MLUploadModelChunkResponse> listener) {
        final var modelId = mlUploadInput.getModelId();
        // Check the size of the content not to exceed 10 mb
        GetRequest getRequest = new GetRequest(ML_MODEL_INDEX).id(modelId);
        try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
            client.get(getRequest, ActionListener.wrap(r -> {
                log.info("Completed Get Model Request, id:{}", modelId);

                if (r != null && r.isExists()) {
                    try (XContentParser parser = createXContentParserFromRegistry(xContentRegistry, r.getSourceAsBytesRef())) {
                        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
                        // Use this model to update the chunk count
                        MLModel existingModel = MLModel.parse(parser);
                        existingModel.setModelId(r.getId());
                        if (existingModel.getTotalChunks() <= mlUploadInput.getChunkNumber()) {
                            throw new Exception("Chunk number exceeds total chunks");
                        }
                        byte[] bytes = mlUploadInput.getContent();
                        if (bytes == null || bytes.length == 0 || isChunckExceeding10MB(bytes.length)) {
                            throw new Exception("Chunk size either 0 or exceeds 10MB");
                        }
                        mlIndicesHandler.initModelIndexIfAbsent(ActionListener.wrap(res -> {
                            int chunkNum = mlUploadInput.getChunkNumber();
                            MLModel mlModel = MLModel
                                .builder()
                                .modelId(existingModel.getModelId())
                                .modelFormat(existingModel.getModelFormat())
                                .totalChunks(existingModel.getTotalChunks())
                                .modelTaskType(existingModel.getModelTaskType())
                                .algorithm(existingModel.getAlgorithm())
                                .chunkNumber(chunkNum)
                                .content(Base64.getEncoder().encodeToString(bytes))// TODO: performance testing to evaluate what limits to
                                                                                   // place on model size
                                .build();
                            IndexRequest indexRequest = new IndexRequest(ML_MODEL_INDEX);
                            indexRequest.id(customModelId(mlUploadInput.getModelId(), mlUploadInput.getChunkNumber()));
                            indexRequest
                                .source(mlModel.toXContent(XContentBuilder.builder(XContentType.JSON.xContent()), ToXContent.EMPTY_PARAMS));
                            indexRequest.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                            client.index(indexRequest, ActionListener.wrap(response -> {
                                log.info("Index model successfully {}, chunk number {}", mlUploadInput.getModelId(), chunkNum + 1);
                                if (existingModel.getTotalChunks() == mlUploadInput.getChunkNumber()) {
                                    Semaphore semaphore = new Semaphore(1);
                                    semaphore.acquire();
                                    MLModel mlModelMeta = MLModel
                                        .builder()
                                        .name(existingModel.getName())
                                        .algorithm(FunctionName.TEXT_EMBEDDING)
                                        .version(existingModel.getVersion())
                                        .modelFormat(existingModel.getModelFormat())
                                        .modelTaskType(MLModelTaskType.TEXT_EMBEDDING)
                                        .modelState(MLModelState.UPLOADED)
                                        .modelConfig(existingModel.getModelConfig())
                                        .totalChunks(existingModel.getTotalChunks())
                                        .modelContentHash(existingModel.getModelContentHash())
                                        .modelContentSizeInBytes(existingModel.getModelContentSizeInBytes())
                                        .createdTime(existingModel.getCreatedTime())
                                        .build();
                                    IndexRequest indexRequest1 = new IndexRequest(ML_MODEL_INDEX);
                                    indexRequest1.id(modelId);
                                    indexRequest1
                                        .source(
                                            mlModelMeta
                                                .toXContent(XContentBuilder.builder(XContentType.JSON.xContent()), ToXContent.EMPTY_PARAMS)
                                        );
                                    indexRequest1.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                                    client.index(indexRequest1, ActionListener.wrap(re -> {
                                        log.debug("Index model successfully {}", existingModel.getName());
                                        semaphore.release();
                                    }, e -> {
                                        log.error("Failed to update model state", e);
                                        semaphore.release();
                                        listener.onFailure(e);
                                    }));
                                }
                                listener.onResponse(new MLUploadModelChunkResponse("Uploaded"));
                            }, e -> {
                                log.error("Failed to upload chunk model", e);
                                listener.onFailure(e);
                            }));
                        }, ex -> {
                            log.error("Failed to init model index", ex);
                            listener.onFailure(ex);
                        }));
                    } catch (Exception e) {
                        log.error("Failed to parse ml model" + r.getId(), e);
                        listener.onFailure(e);
                    }
                } else {
                    listener.onFailure(new MLResourceNotFoundException("Failed to find model"));
                }
            }, e -> {
                if (e instanceof IndexNotFoundException) {
                    listener.onFailure(new MLResourceNotFoundException("Failed to find model"));
                } else {
                    log.error("Failed to get ML model " + modelId, e);
                    listener.onFailure(e);
                }
            }));
        } catch (Exception e) {
            log.error(e.getMessage());
            listener.onFailure(e);
        }
    }

    public static String customModelId(String modelId, Integer chunkNumber) {
        return modelId + "_" + chunkNumber;
    }

    private boolean isChunckExceeding10MB(final long length) {
        var isChunkExceedsSize = false;
        double fileSizeInKB = length / 1024;
        double fileSizeInMB = fileSizeInKB / 1024;
        if (fileSizeInMB > 10.0d) {
            isChunkExceedsSize = true;
        }
        return isChunkExceedsSize;
    }
}
