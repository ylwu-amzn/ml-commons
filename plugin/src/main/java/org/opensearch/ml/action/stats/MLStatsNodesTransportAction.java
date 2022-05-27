/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.stats;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opensearch.action.FailedNodeException;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.nodes.TransportNodesAction;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.env.Environment;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.stats.ActionName;
import org.opensearch.ml.stats.MLStats;
import org.opensearch.monitor.jvm.JvmService;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

public class MLStatsNodesTransportAction extends
    TransportNodesAction<MLStatsNodesRequest, MLStatsNodesResponse, MLStatsNodeRequest, MLStatsNodeResponse> {
    private MLStats mlStats;
    private final JvmService jvmService;

    /**
     * Constructor
     *
     * @param threadPool ThreadPool to use
     * @param clusterService ClusterService
     * @param transportService TransportService
     * @param actionFilters Action Filters
     * @param mlStats MLStats object
     * @param environment OpenSearch Environment
     */
    @Inject
    public MLStatsNodesTransportAction(
        ThreadPool threadPool,
        ClusterService clusterService,
        TransportService transportService,
        ActionFilters actionFilters,
        MLStats mlStats,
        Environment environment
    ) {
        super(
            MLStatsNodesAction.NAME,
            threadPool,
            clusterService,
            transportService,
            actionFilters,
            MLStatsNodesRequest::new,
            MLStatsNodeRequest::new,
            ThreadPool.Names.MANAGEMENT,
            MLStatsNodeResponse.class
        );
        this.mlStats = mlStats;
        this.jvmService = new JvmService(environment.settings());
    }

    @Override
    protected MLStatsNodesResponse newResponse(
        MLStatsNodesRequest request,
        List<MLStatsNodeResponse> responses,
        List<FailedNodeException> failures
    ) {
        return new MLStatsNodesResponse(clusterService.getClusterName(), responses, failures);
    }

    @Override
    protected MLStatsNodeRequest newNodeRequest(MLStatsNodesRequest request) {
        return new MLStatsNodeRequest(request);
    }

    @Override
    protected MLStatsNodeResponse newNodeResponse(StreamInput in) throws IOException {
        return new MLStatsNodeResponse(in);
    }

    @Override
    protected MLStatsNodeResponse nodeOperation(MLStatsNodeRequest request) {
        return createMLStatsNodeResponse(request.getMlStatsNodesRequest());
    }

    private MLStatsNodeResponse createMLStatsNodeResponse(MLStatsNodesRequest mlStatsNodesRequest) {
        Map<MLNodeLevelStat, Object> statValues = new HashMap<>();
        MLStatsInput mlStatsInput = mlStatsNodesRequest.getMlStatsInput();
        // return node level stats
        if (mlStatsInput.getTargetStatLevels().contains(MLStatLevel.NODE)) {
            boolean retrieveAllNodeStats = mlStatsInput.getNodeLevelStats().size() == 0;
            if (retrieveAllNodeStats || mlStatsInput.getNodeLevelStats().contains(MLNodeLevelStat.ML_NODE_JVM_HEAP_USAGE)) {
                long heapUsedPercent = jvmService.stats().getMem().getHeapUsedPercent();
                statValues.put(MLNodeLevelStat.ML_NODE_JVM_HEAP_USAGE, heapUsedPercent);
            }

            for (Enum statName : mlStats.getNodeStats().keySet()) {
                if (retrieveAllNodeStats || mlStatsInput.getNodeLevelStats().contains(statName)) {
                    statValues.put((MLNodeLevelStat) statName, mlStats.getStats().get(statName).getValue());
                }
            }
        }

        Map<FunctionName, MLAlgoActionStats> algorithmStats = new HashMap<>();
        // return algorithm level stats
        if (mlStatsInput.getTargetStatLevels().contains(MLStatLevel.ALGORITHM)) {
            boolean retrieveAllAlgoStats = mlStatsInput.getAlgorithms().size() == 0;
            for (FunctionName algoName : mlStats.getAllAlgorithms()) {
                if (retrieveAllAlgoStats || mlStatsInput.getAlgorithms().contains(algoName)) {
                    Map<ActionName, MLAlgoStats> algoActionStatsMap = mlStats.getAlgorithmStats(algoName);
                    algorithmStats.put(algoName, new MLAlgoActionStats(algoActionStatsMap));
                }
            }
        }

        return new MLStatsNodeResponse(clusterService.localNode(), statValues, algorithmStats);
    }
}
