package org.opensearch.ml.action.custom.uploadchunk;

import lombok.extern.log4j.Log4j2;
import org.opensearch.action.ActionListener;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.client.Client;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.Model;
import org.opensearch.ml.common.transport.custom.load.LoadModelResponse;
import org.opensearch.ml.common.transport.custom.uploadchunk.MLUploadChunkInput;
import org.opensearch.ml.engine.algorithms.custom.CustomModelManager;
import org.opensearch.ml.indices.MLIndicesHandler;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.threadpool.ThreadPool;

import java.util.Base64;

import static org.opensearch.ml.common.CommonValue.ML_MODEL_INDEX;

@Log4j2
public class MLModelChunkUploader {

    public static final int TIMEOUT_IN_MILLIS = 5000;
    public static final int MAX_ACCEPTED_CHUNK_SIZE_STRING_LEN = 100000000;  /* 100MB */
    private final CustomModelManager customModelManager;
    private final MLIndicesHandler mlIndicesHandler;
    private final MLTaskManager mlTaskManager;
    private final ThreadPool threadPool;
    private final Client client;

    public MLModelChunkUploader(CustomModelManager customModelManager, MLIndicesHandler mlIndicesHandler, MLTaskManager mlTaskManager, ThreadPool threadPool, Client client) {
        this.customModelManager = customModelManager;
        this.mlIndicesHandler = mlIndicesHandler;
        this.mlTaskManager = mlTaskManager;
        this.threadPool = threadPool;
        this.client = client;
    }

    public void uploadModel(MLUploadChunkInput mlUploadInput, ActionListener<LoadModelResponse> listener) {

        try {
            String modelName = mlUploadInput.getName(); // get name of model
            Integer version = mlUploadInput.getVersion(); // get version of model
            mlIndicesHandler.initModelIndexIfAbsent(ActionListener.wrap(res -> {
                byte[] bytes = mlUploadInput.getUrl();
                Model model = new Model();
                model.setName(FunctionName.CUSTOM.name());
                model.setVersion(1);
                model.setContent(bytes);
                int chunkNum = mlUploadInput.getChunkNumber();
                int totalChunks = mlUploadInput.getTotalChunks();
                MLModel mlModel = MLModel.builder()
                        .name(modelName)
                        .algorithm(FunctionName.CUSTOM)
                        .version(version)
                        .chunkNumber(chunkNum)
                        .totalChunks(totalChunks)
                        .content(Base64.getEncoder().encodeToString(bytes))
                        .build();
                IndexRequest indexRequest = new IndexRequest(ML_MODEL_INDEX);
                indexRequest.id(MLModel.customModelId(modelName, version, chunkNum));//TODO: limit model name size and not include "_"
                indexRequest.source(mlModel.toXContent(XContentBuilder.builder(XContentType.JSON.xContent()), ToXContent.EMPTY_PARAMS));
                indexRequest.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);

                client.index(indexRequest, ActionListener.wrap(r -> {
                    log.info("Index model successfully {}, chunk number {} out of {}", modelName, chunkNum, totalChunks);
                    listener.onResponse(new LoadModelResponse("0", "1"));
                }, e -> {
                    log.error("Failed to index model", e);
                    listener.onFailure(e);
                }));
            }, ex -> {
                log.error("Failed to init model index", ex);
                listener.onFailure(ex);
            }));
        } catch (Exception e) {
            log.error("Failed to upload model ", e);
            listener.onFailure(e);
        }
    }
}
