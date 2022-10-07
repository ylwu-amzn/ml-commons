/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.model;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.common.CommonValue.DELETED;
import static org.opensearch.ml.common.CommonValue.ML_MODEL_INDEX;
import static org.opensearch.ml.common.CommonValue.NOT_FOUND;
import static org.opensearch.ml.engine.MLEngine.getLoadModelChunkPath;
import static org.opensearch.ml.engine.MLEngine.getLoadModelZipPath;
import static org.opensearch.ml.plugin.MachineLearningPlugin.TASK_THREAD_POOL;
import static org.opensearch.ml.settings.MLCommonsSettings.ML_COMMONS_MAX_MODELS_PER_NODE;
import static org.opensearch.ml.utils.MLNodeUtils.createXContentParserFromRegistry;

import java.io.File;
import java.nio.file.Path;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.action.update.UpdateRequest;
import org.opensearch.action.update.UpdateResponse;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.exception.MLResourceNotFoundException;
import org.opensearch.ml.common.model.MLModelFormat;
import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.common.transport.custom_model.unload.UnloadModelInput;
import org.opensearch.ml.engine.MLEngine;
import org.opensearch.ml.engine.Predictable;
import org.opensearch.ml.engine.algorithms.custom.CustomModelManager;
import org.opensearch.ml.engine.utils.MLFileUtils;
import org.opensearch.rest.RestStatus;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.threadpool.ThreadPool;

import com.google.common.collect.ImmutableMap;

@Log4j2
public class MLModelManager {

    private final Client client;
    private ThreadPool threadPool;
    private NamedXContentRegistry xContentRegistry;
    private CustomModelManager customModelManager;

    private final MLModelCache modelCache;
    private volatile Integer maxModelPerNode;

    public MLModelManager(
        ClusterService clusterService,
        Client client,
        ThreadPool threadPool,
        NamedXContentRegistry xContentRegistry,
        CustomModelManager customModelManager,
        Settings settings
    ) {
        this.client = client;
        this.threadPool = threadPool;
        this.xContentRegistry = xContentRegistry;
        this.customModelManager = customModelManager;
        this.modelCache = new MLModelCache();
        this.maxModelPerNode = ML_COMMONS_MAX_MODELS_PER_NODE.get(settings);
        clusterService.getClusterSettings().addSettingsUpdateConsumer(ML_COMMONS_MAX_MODELS_PER_NODE, it -> maxModelPerNode = it);
    }

    public void loadModel1(String modelId, ActionListener<String> listener) {
        if (modelCache.isModelLoaded(modelId)) {
            listener.onResponse("successful");
            return;
        }
        if (modelCache.modelCount() >= maxModelPerNode) {
            listener.onFailure(new IllegalArgumentException("Exceed max model per node limit"));
            return;
        }
        modelCache.initModelState(modelId, MLModelState.LOADING);
        try {
            threadPool.executor(TASK_THREAD_POOL).execute(() -> {
                try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                    this.getModel(modelId, ActionListener.wrap(mlModelMeta -> {
                        if (mlModelMeta.getAlgorithm() != FunctionName.CUSTOM) {// load model trained by built-in algorithm like kmeans
                            Predictable predictable = MLEngine.load(mlModelMeta);
                            modelCache.addPredictable(modelId, predictable);
                            modelCache.setModelState(modelId, MLModelState.LOADED);
                            listener.onResponse("successful");
                            return;
                        }
                        String engine = mlModelMeta.getModelFormat() == MLModelFormat.TORCH_SCRIPT ? "PyTorch" : "OnnxRuntime";
                        retrieveModelChunks(mlModelMeta, ActionListener.wrap(modelZipFile -> {
                            customModelManager
                                .loadModel(
                                    modelZipFile,
                                    modelId,
                                    mlModelMeta.getName(),
                                    mlModelMeta.getModelTaskType(),
                                    mlModelMeta.getVersion(),
                                    mlModelMeta.getModelConfig(),
                                    engine
                                );
                            modelCache.setModelState(modelId, MLModelState.LOADED);
                            listener.onResponse("successful");
                        }, e -> {
                            e.printStackTrace();
                            log.error("Failed to retrieve model " + modelId, e);
                            modelCache.removeModelState(modelId);
                            listener.onFailure(e);
                        }));
                    }, e -> {
                        modelCache.removeModelState(modelId);
                        listener.onFailure(new MLResourceNotFoundException("ML model not found"));
                    }));
                } catch (Exception e) {
                    modelCache.removeModelState(modelId);
                    listener.onFailure(e);
                }
            });
        } catch (Exception e) {
            modelCache.removeModelState(modelId);
            listener.onFailure(e);
        }
    }

    public void getModel(String modelId, ActionListener<MLModel> listener) {
        GetRequest getRequest = new GetRequest();
        getRequest.index(ML_MODEL_INDEX).id(modelId);
        client.get(getRequest, ActionListener.wrap(r -> {
            if (r != null && r.isExists()) {
                try (XContentParser parser = createXContentParserFromRegistry(xContentRegistry, r.getSourceAsBytesRef())) {
                    ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
                    MLModel mlModel = MLModel.parse(parser);
                    mlModel.setModelId(modelId);
                    listener.onResponse(mlModel);
                } catch (Exception e) {
                    log.error("Failed to parse ml task" + r.getId(), e);
                    listener.onFailure(e);
                }
            } else {
                listener.onFailure(new MLResourceNotFoundException("Fail to find model"));
            }
        }, e -> { listener.onFailure(e); }));
    }

    public void getModel(String modelId, String[] includes, String[] excludes, ActionListener<MLModel> listener) {
        GetRequest getRequest = new GetRequest();
        FetchSourceContext featchContext = new FetchSourceContext(true, includes, excludes);
        getRequest.index(ML_MODEL_INDEX).id(modelId).fetchSourceContext(featchContext);
        client.get(getRequest, ActionListener.wrap(r -> {
            if (r != null && r.isExists()) {
                try (XContentParser parser = createXContentParserFromRegistry(xContentRegistry, r.getSourceAsBytesRef())) {
                    ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
                    MLModel mlModel = MLModel.parse(parser);
                    mlModel.setModelId(modelId);
                    listener.onResponse(mlModel);
                } catch (Exception e) {
                    log.error("Failed to parse ml task" + r.getId(), e);
                    listener.onFailure(e);
                }
            } else {
                listener.onFailure(new MLResourceNotFoundException("Fail to find model"));
            }
        }, e -> { listener.onFailure(e); }));
    }

    private void retrieveModelChunks(MLModel mlModelMeta, ActionListener<File> listener) throws InterruptedException {
        String modelId = mlModelMeta.getModelId();
        String modelName = mlModelMeta.getName();
        Integer totalChunks = mlModelMeta.getTotalChunks();
        GetRequest getRequest = new GetRequest();
        getRequest.index(ML_MODEL_INDEX);
        getRequest.id();
        Semaphore semaphore = new Semaphore(1);
        AtomicBoolean stopNow = new AtomicBoolean(false);
        String modelZip = getLoadModelZipPath(modelId, modelName);
        File[] chunkFiles = new File[totalChunks];
        AtomicInteger retrievedChunks = new AtomicInteger(0);
        for (int i = 0; i < totalChunks; i++) {
            if (stopNow.get()) {
                listener.onFailure(new MLException("Failed to load model"));
                return;
            }
            semaphore.tryAcquire(10, TimeUnit.SECONDS);

            String modelChunkId = this.getModelChunkId(modelId, i);
            int currentChunk = i;
            this.getModel(modelChunkId, ActionListener.wrap(model -> {
                Path chunkPath = getLoadModelChunkPath(modelId, currentChunk);
                MLFileUtils.write(Base64.getDecoder().decode(model.getContent()), chunkPath.toString());
                chunkFiles[currentChunk] = new File(chunkPath.toUri());
                semaphore.release();
                retrievedChunks.getAndIncrement();
                if (retrievedChunks.get() == totalChunks) {
                    File modelZipFile = new File(modelZip);
                    MLFileUtils.mergeFiles(chunkFiles, modelZipFile);
                    listener.onResponse(modelZipFile);
                }
            }, e -> {
                e.printStackTrace();
                stopNow.set(true);
                semaphore.release();
                listener.onFailure(new MLResourceNotFoundException("Fail to find model chunk"));
                return;
            }));
        }
    }

    public void updateModel(String modelId, ImmutableMap<String, Object> updatedFields) {
        updateModel(modelId, updatedFields, ActionListener.wrap(response -> {
            if (response.status() == RestStatus.OK) {
                log.debug("Updated ML model successfully: {}, model id: {}", response.status(), modelId);
            } else {
                log.error("Failed to update ML model {}, status: {}", modelId, response.status());
            }
        }, e -> { log.error("Failed to update ML model: " + modelId, e); }));
    }

    public void updateModel(String modelId, ImmutableMap<String, Object> updatedFields, ActionListener<UpdateResponse> listener) {
        try {
            if (updatedFields == null || updatedFields.size() == 0) {
                listener.onFailure(new IllegalArgumentException("Updated fields is null or empty"));
                return;
            }
            UpdateRequest updateRequest = new UpdateRequest(ML_MODEL_INDEX, modelId);
            updateRequest.doc(updatedFields);
            updateRequest.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                client.update(updateRequest, ActionListener.runBefore(listener, () -> context.restore()));
            } catch (Exception e) {
                listener.onFailure(e);
            }
        } catch (Exception e) {
            log.error("Failed to update ML model " + modelId, e);
            listener.onFailure(e);
        }
    }

    public String getModelChunkId(String modelId, Integer chunkNumber) {
        return modelId + "_" + chunkNumber;
    }

    public void addModelWorkerNode(String modelId, String... nodeIds) {
        if (nodeIds != null) {
            for (String nodeId : nodeIds) {
                modelCache.addModelWorkerNode(modelId, nodeId);
            }
        }
    }

    public void removeModelWorkerNode(String modelId, String... nodeIds) {
        if (nodeIds != null) {
            for (String nodeId : nodeIds) {
                modelCache.removeModelWorkerNode(modelId, nodeId);
            }
        }
    }

    public void removeWorkerNodes(Set<String> removedNodes) {
        modelCache.removeWorkNodes(removedNodes);
    }

    public synchronized Map<String, String> unloadModel(UnloadModelInput unloadModelInput) {
        Map<String, String> modelUnloadStatus = new HashMap<>();
        String[] modelIds = unloadModelInput.getModelIds();
        String[] nodeIds = unloadModelInput.getNodeIds();
        if (modelIds != null && modelIds.length > 0) {
            for (String modelId : modelIds) {
                if (modelCache.hasModel(modelId)) {
                    modelUnloadStatus.put(modelId, DELETED);
                } else {
                    modelUnloadStatus.put(modelId, NOT_FOUND);
                }
                modelCache.removeModel(modelId, nodeIds);
            }
        }
        return modelUnloadStatus;
    }

    public String[] getWorkerNodes(String modelId) {
        return modelCache.getWorkerNodes(modelId);
    }

    public void addPredictable(String modelId, Predictable predictable) {
        modelCache.addPredictable(modelId, predictable);
    }

    public Predictable getPredictable(String modelId) {
        return modelCache.getPredictable(modelId);
    }

    public String[] getLocalLoadedModels() {
        String[] loadedModels = modelCache.getLoadedModels();
        String[] localLoadedModels = customModelManager.getLocalLoadedModels();
        String[] result = new String[loadedModels.length + localLoadedModels.length];
        int i = 0;
        for (; i < loadedModels.length; i++) {
            result[i] = loadedModels[i];
        }
        for (int j = 0; j < localLoadedModels.length; j++) {
            result[i + j] = localLoadedModels[j];
        }
        return result;
    }

    public synchronized void syncModelRouting(Map<String, Set<String>> modelRoutingTable) {
        modelCache.syncModelRouting(modelRoutingTable);
    }
}
