/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.model;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import lombok.extern.log4j.Log4j2;

import org.opensearch.ml.common.model.MLModelState;

@Log4j2
public class MLModelCache {

    private final Map<String, MLModelState> modelStates;
    private final Map<String, Set<String>> modelWorkerNodes;

    public MLModelCache() {
        this.modelStates = new ConcurrentHashMap<>();
        this.modelWorkerNodes = new ConcurrentHashMap<>();
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
        Set<String> nodes = modelWorkerNodes.get(modelId);
        if (nodes != null) {
            nodes.removeAll(removedNodes);
        }
    }

    public void removeWorkNodes(Set<String> removedNodes) {
        for (Map.Entry<String, Set<String>> entry : modelWorkerNodes.entrySet()) {
            Set<String> nodes = entry.getValue();
            nodes.removeAll(removedNodes);
        }
    }

    public synchronized void addModelWorkerNode(String modelId, String nodeId) {
        if (!modelWorkerNodes.containsKey(modelId)) {
            ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
            Set<String> set = map.newKeySet();
            modelWorkerNodes.put(modelId, set);
        }
        modelWorkerNodes.get(modelId).add(nodeId);
    }

    public synchronized void removeModelWorkerNode(String modelId, String nodeId) {
        if (!modelWorkerNodes.containsKey(modelId)) {
            return;
        }
        modelWorkerNodes.get(modelId).remove(nodeId);
    }

    public void removeModel(String modelId) {
        this.modelStates.remove(modelId);
        this.modelWorkerNodes.remove(modelId);
    }

    public String[] getWorkerNodes(String modelId) {
        Set<String> nodes = modelWorkerNodes.get(modelId);
        if (nodes == null) {
            return null;
        }
        return nodes.toArray(new String[0]);
    }
}
