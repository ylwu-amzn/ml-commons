/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.model;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import lombok.extern.log4j.Log4j2;

import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.engine.Predictable;

@Log4j2
public class MLModelCache {

    private final Map<String, MLModelState> modelStates;
    private final Map<String, Predictable> predictors;
    private final Map<String, Set<String>> modelRoutingTable;// routingTable

    public MLModelCache() {
        this.modelStates = new ConcurrentHashMap<>();
        this.predictors = new ConcurrentHashMap<>();
        this.modelRoutingTable = new ConcurrentHashMap<>();
    }

    public synchronized boolean hasModel(String modelId) {
        return predictors.containsKey(modelId);
    }

    public synchronized boolean isModelLoaded(String modelId) {
        MLModelState mlModelState = modelStates.get(modelId);
        if (mlModelState == MLModelState.LOADED) {
            return true;
        }
        return false;
    }

    public synchronized void initModelState(String modelId, MLModelState state) {
        if (modelStates.containsKey(modelId)) {
            throw new IllegalArgumentException("Duplicate model task");
        }
        modelStates.put(modelId, state);
    }

    public synchronized void setModelState(String modelId, MLModelState state) {
        if (!modelStates.containsKey(modelId)) {
            throw new IllegalArgumentException("Model not found in cache");
        }
        modelStates.put(modelId, state);
    }

    public void removeModelState(String modelId) {
        modelStates.remove(modelId);
    }

    public void removeModelWorkerNode(String modelId, Set<String> removedNodes) {
        Set<String> nodes = modelRoutingTable.get(modelId);
        if (nodes != null) {
            nodes.removeAll(removedNodes);
        }
    }

    public void removeWorkNodes(Set<String> removedNodes) {
        for (Map.Entry<String, Set<String>> entry : modelRoutingTable.entrySet()) {
            Set<String> nodes = entry.getValue();
            nodes.removeAll(removedNodes);
        }
    }

    public synchronized void addPredictable(String modelId, Predictable predictable) {
        this.predictors.put(modelId, predictable);
    }

    public synchronized void addModelWorkerNode(String modelId, String nodeId) {
        if (!modelRoutingTable.containsKey(modelId)) {
            ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
            Set<String> set = map.newKeySet();
            modelRoutingTable.put(modelId, set);
        }
        log.debug("add node {} to model cache for model {}", nodeId, modelId);
        modelRoutingTable.get(modelId).add(nodeId);
    }

    public synchronized void removeModelWorkerNode(String modelId, String nodeId) {
        if (!modelRoutingTable.containsKey(modelId)) {
            log.info("model {} not found in cache", modelId);
            return;
        }
        log.debug("remove node {} from model cache for model {}", nodeId, modelId);
        modelRoutingTable.get(modelId).remove(nodeId);
        if (modelRoutingTable.get(modelId).size() == 0) {
            log.info("remove model {} from worker node cache as it's worker node size is 0", modelId);
            modelRoutingTable.remove(modelId);
        }
    }

    public void removeModel(String modelId, String[] nodeIds) {
        this.modelStates.remove(modelId);
        this.predictors.remove(modelId);
        log.debug("remove model state and predictable model {}", modelId);
        if (nodeIds == null || nodeIds.length == 0) {
            this.modelRoutingTable.remove(modelId);
        } else {
            for (String nodeId : nodeIds) {
                removeModelWorkerNode(modelId, nodeId);
            }
        }
    }

    public String[] getWorkerNodes(String modelId) {
        Set<String> nodes = modelRoutingTable.get(modelId);
        if (nodes == null) {
            return null;
        }
        return nodes.toArray(new String[0]);
    }

    public Predictable getPredictable(String modelId) {
        return predictors.get(modelId);
    }

    public synchronized int modelCount() {
        return modelStates.size();
    }

    public String[] getLoadedModels() {
        return predictors.keySet().toArray(new String[0]);
    }

    public void syncModelRouting(Map<String, Set<String>> modelRoutingTable) {
        log.debug("sync model routing for model");
        Set<String> currentModels = new HashSet(this.modelRoutingTable.keySet());
        this.modelRoutingTable.putAll(modelRoutingTable);
        currentModels.removeAll(modelRoutingTable.keySet());
        if (currentModels.size() > 0) {
            currentModels.forEach(k -> this.modelRoutingTable.remove(k));
        }
    }
}
