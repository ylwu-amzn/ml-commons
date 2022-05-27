/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.stats;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

import lombok.Getter;

import org.opensearch.ml.action.stats.MLAlgoStats;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.stats.suppliers.CounterSupplier;

/**
 * This class is the main entry-point for access to the stats that the ML plugin keeps track of.
 */
public class MLStats {
    @Getter
    private Map<String, MLStat<?>> stats;
    private Map<FunctionName, Map<ActionName, Map<String, MLStat>>> algoStats;// {"kmeans":{"train":{"request_count":10}}}

    /**
     * Constructor
     *
     * @param stats Map of the stats that are to be kept
     */
    public MLStats(Map<String, MLStat<?>> stats) {
        this.stats = stats;
        this.algoStats = new ConcurrentHashMap<>();
    }

    /**
     * Get individual stat by stat name
     *
     * @param key Name of stat
     * @return ADStat
     * @throws IllegalArgumentException thrown on illegal statName
     */
    public MLStat<?> getStat(String key) throws IllegalArgumentException {
        if (!stats.keySet().contains(key)) {
            throw new IllegalArgumentException("Stat \"" + key + "\" does not exist");
        }
        return stats.get(key);
    }

    /**
     * Get stat or create counter stat if absent.
     * @param key stat key
     * @return existing MLStat or new MLStat
     */
    public MLStat<?> createCounterStatIfAbsent(String key) {
        return createStatIfAbsent(key, () -> new MLStat<>(false, new CounterSupplier()));
    }

    public MLStat<?> createCounterStatIfAbsent(FunctionName algoName, ActionName action, String statKey) {
        Map<ActionName, Map<String, MLStat>> actionStats = algoStats.computeIfAbsent(algoName, it -> new ConcurrentHashMap<>());
        Map<String, MLStat> algoActionStats = actionStats.computeIfAbsent(action, it -> new ConcurrentHashMap<>());
        return createAlgoStatIfAbsent(algoActionStats, statKey, () -> new MLStat<>(false, new CounterSupplier()));
    }

    public synchronized MLStat<?> createAlgoStatIfAbsent(Map<String, MLStat> algoActionStats, String key, Supplier<MLStat> supplier) {
        return algoActionStats.computeIfAbsent(key, k -> supplier.get());
    }

    /**
     * Get stat or create if absent.
     * @param key stat key
     * @param supplier supplier to create MLStat
     * @return existing MLStat or new MLStat
     */
    public synchronized MLStat<?> createStatIfAbsent(String key, Supplier<MLStat> supplier) {
        return stats.computeIfAbsent(key, k -> supplier.get());
    }

    /**
     * Get a map of the stats that are kept at the node level
     *
     * @return Map of stats kept at the node level
     */
    public Map<String, MLStat<?>> getNodeStats() {
        return getClusterOrNodeStats(false);
    }

    /**
     * Get a map of the stats that are kept at the cluster level
     *
     * @return Map of stats kept at the cluster level
     */
    public Map<String, MLStat<?>> getClusterStats() {
        return getClusterOrNodeStats(true);
    }

    private Map<String, MLStat<?>> getClusterOrNodeStats(Boolean getClusterStats) {
        Map<String, MLStat<?>> statsMap = new HashMap<>();

        for (Map.Entry<String, MLStat<?>> entry : stats.entrySet()) {
            if (entry.getValue().isClusterLevel() == getClusterStats) {
                statsMap.put(entry.getKey(), entry.getValue());
            }
        }
        return statsMap;
    }

    // public Map<FunctionName, Map<ActionName, Map<String, MLStat>>> getAllAlgorithmStats() {
    // Map<FunctionName, Map<ActionName, Map<String, MLStat>>> algoStats = new HashMap<>();
    // for (FunctionName algoName : algoStats.keySet()) {
    // Map<ActionName, Map<String, MLStat>> stats = getAlgorithmStats(algoName);
    // if (stats != null) {
    // algoStats.put(algoName, stats);
    // }
    // }
    // return algoStats;
    // }

    // public Map<ActionName, Map<String, MLStat>> getAlgorithmStats(FunctionName algoName) {
    // if (!algoStats.containsKey(algoName)) {
    // return null;
    // }
    // Map<ActionName, Map<String, MLStat>> algoActionStats = new HashMap<>();
    //
    // for (Map.Entry<ActionName, Map<String, MLStat>> entry : algoStats.get(algoName).entrySet()) {
    // Map<String, MLStat> stats = new HashMap<>();
    // for (Map.Entry<String, MLStat> state: entry.getValue().entrySet()) {
    // stats.put(state.getKey(), state.getValue());
    // }
    // algoActionStats.put(entry.getKey(), stats);
    // }
    // return algoActionStats;
    // }

    public Map<String, MLAlgoStats> getAlgorithmStats(FunctionName algoName) {
        if (!algoStats.containsKey(algoName)) {
            return null;
        }
        Map<String, MLAlgoStats> algoActionStats = new HashMap<>();

        for (Map.Entry<ActionName, Map<String, MLStat>> entry : algoStats.get(algoName).entrySet()) {
            Map<String, Object> statsMap = new HashMap<>();
            for (Map.Entry<String, MLStat> state : entry.getValue().entrySet()) {
                statsMap.put(state.getKey(), state.getValue().getValue());
            }
            algoActionStats.put(entry.getKey().name(), new MLAlgoStats(statsMap));
        }
        return algoActionStats;
    }

    public FunctionName[] getAllAlgorithms() {
        return algoStats.keySet().toArray(new FunctionName[0]);
    }
}
