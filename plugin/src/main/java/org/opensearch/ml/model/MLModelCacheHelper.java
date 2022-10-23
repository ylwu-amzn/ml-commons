/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.model;

import static org.opensearch.ml.settings.MLCommonsSettings.ML_COMMONS_MONITORING_REQUEST_COUNT;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import lombok.extern.log4j.Log4j2;

import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.engine.Predictable;
import org.opensearch.ml.profile.MLModelProfile;
import org.opensearch.ml.profile.MLPredictRequestStats;

@Log4j2
public class MLModelCacheHelper {
    private final Map<String, MLModelCache> modelCaches;
    private volatile Long maxRequestCount;

    public MLModelCacheHelper(ClusterService clusterService, Settings settings) {
        this.modelCaches = new ConcurrentHashMap<>();

        maxRequestCount = ML_COMMONS_MONITORING_REQUEST_COUNT.get(settings);
        clusterService.getClusterSettings().addSettingsUpdateConsumer(ML_COMMONS_MONITORING_REQUEST_COUNT, it -> maxRequestCount = it);
    }

    public synchronized void initModelState(String modelId, MLModelState state, FunctionName functionName) {
        if (modelCaches.containsKey(modelId)) {
            throw new IllegalArgumentException("Duplicate model task");
        }
        MLModelCache modelCache = new MLModelCache();
        modelCache.setModelState(state);
        modelCache.setFunctionName(functionName);
        modelCaches.put(modelId, modelCache);
    }

    public synchronized void setModelState(String modelId, MLModelState state) {
        getExistingModelCache(modelId).setModelState(state);
    }

    public synchronized boolean isModelLoaded(String modelId) {
        MLModelCache modelCache = modelCaches.get(modelId);
        return modelCache != null && modelCache.getModelState() == MLModelState.LOADED;
    }

    public synchronized void addPredictor(String modelId, Predictable predictable) {
        MLModelCache modelCache = getExistingModelCache(modelId);
        modelCache.setPredictor(predictable);
    }

    public Predictable getPredictor(String modelId) {
        MLModelCache modelCache = modelCaches.get(modelId);
        if (modelCache == null) {
            return null;
        }
        return modelCache.getPredictor();
    }

    public void removeModel(String modelId) {
        MLModelCache modelCache = modelCaches.get(modelId);
        if (modelCache != null) {
            log.debug("remove model from cache: {}", modelId);
            modelCache.clear();
            modelCaches.remove(modelId);
        }
    }

    public String[] getLoadedModels() {
        return modelCaches
            .entrySet()
            .stream()
            .filter(entry -> entry.getValue().getModelState() == MLModelState.LOADED)
            .map(entry -> entry.getKey())
            .collect(Collectors.toList())
            .toArray(new String[0]);
    }

    public String[] getAllModels() {
        return modelCaches.keySet().toArray(new String[0]);
    }

    public String[] getWorkerNodes(String modelId) {
        MLModelCache modelCache = modelCaches.get(modelId);
        if (modelCache == null) {
            return null;
        }
        return modelCache.getWorkerNodes();
    }

    public synchronized void addWorkerNode(String modelId, String nodeId) {
        log.debug("add node {} to model routing table for model: {}", nodeId, modelId);
        MLModelCache modelCache = getOrCreateModelCache(modelId);
        modelCache.addWorkerNode(nodeId);
    }

    public void removeWorkerNodes(Set<String> removedNodes) {
        Set<String> modelIds = modelCaches.keySet();
        for (String modelId : modelIds) {
            MLModelCache modelCache = modelCaches.get(modelId);
            log.debug("remove worker nodes of model {} : {}", modelId, removedNodes.toArray(new String[0]));
            modelCache.removeWorkerNodes(removedNodes);
            if (!modelCache.isValidCache()) {
                modelCaches.remove(modelId);
            }
        }
    }

    public void removeWorkerNode(String modelId, String nodeId) {
        MLModelCache modelCache = modelCaches.get(modelId);
        if (modelCache != null) {
            log.debug("remove worker node {} of model {} from cache", nodeId, modelId);
            modelCache.removeWorkerNode(nodeId);
            if (!modelCache.isValidCache()) {
                log.debug("remove model {} from cache as no node running it", modelId);
                modelCaches.remove(modelId);
            }
        }
    }

    public void syncWorkerNodes(Map<String, Set<String>> modelWorkerNodes) {
        log.debug("sync model worker nodes");
        Set<String> currentModels = new HashSet(this.modelCaches.keySet());
        currentModels.removeAll(modelWorkerNodes.keySet());
        if (currentModels.size() > 0) {
            currentModels.forEach(modelId -> { clearWorkerNodes(modelId); });
        }
        modelWorkerNodes.entrySet().forEach(entry -> {
            MLModelCache modelCache = modelCaches.get(entry.getKey());
            if (modelCache != null) {
                modelCache.syncWorkerNode(entry.getValue());
            }
        });
    }

    public void clearWorkerNodes() {
        log.debug("clear all model worker nodes");
        modelCaches.entrySet().forEach(entry -> { clearWorkerNodes(entry.getKey()); });
    }

    public void clearWorkerNodes(String modelId) {
        MLModelCache modelCache = modelCaches.get(modelId);
        if (modelCache != null) {
            log.debug("clear worker nodes of model {}", modelId);
            modelCache.clearWorkerNodes();
            if (!modelCache.isValidCache()) {
                modelCaches.remove(modelId);
            }
        }
    }

    public MLModelProfile getModelProfile(String modelId) {
        MLModelCache modelCache = modelCaches.get(modelId);
        if (modelCache == null) {
            return null;
        }

        MLModelProfile.MLModelProfileBuilder builder = MLModelProfile.builder();
        builder.modelState(modelCache.getModelState());
        if (modelCache.getPredictor() != null) {
            builder.predictor(modelCache.getPredictor().toString());
        }
        String[] workerNodes = modelCache.getWorkerNodes();
        if (workerNodes.length > 0) {
            builder.workerNodes(workerNodes);
        }
        MLPredictRequestStats stats = modelCache.getInferenceStats();
        builder.predictStats(stats);
        return builder.build();
    }

    public void addInferenceDuration(String modelId, double duration) {
        MLModelCache modelCache = getOrCreateModelCache(modelId);
        modelCache.addInferenceDuration(duration, maxRequestCount);
    }

    public FunctionName getFunctionName(String modelId) {
        MLModelCache modelCache = getExistingModelCache(modelId);
        return modelCache.getFunctionName();
    }

    public boolean containsModel(String modelId) {
        MLModelCache modelCache = modelCaches.get(modelId);
        return modelCache != null && modelCache.getModelState() != null;
    }

    private MLModelCache getExistingModelCache(String modelId) {
        MLModelCache modelCache = modelCaches.get(modelId);
        if (modelCache == null) {
            throw new IllegalArgumentException("Model not found in cache");
        }
        return modelCache;
    }

    private MLModelCache getOrCreateModelCache(String modelId) {
        return modelCaches.computeIfAbsent(modelId, it -> new MLModelCache());
    }

}
