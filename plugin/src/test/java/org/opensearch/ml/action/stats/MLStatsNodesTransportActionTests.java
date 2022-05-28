/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.stats;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.IOException;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.junit.Assert;
import org.junit.Before;
import org.opensearch.Version;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.settings.Settings;
import org.opensearch.env.Environment;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.stats.ActionName;
import org.opensearch.ml.stats.MLActionLevelStat;
import org.opensearch.ml.stats.MLActionStats;
import org.opensearch.ml.stats.MLAlgoStats;
import org.opensearch.ml.stats.MLClusterLevelStat;
import org.opensearch.ml.stats.MLNodeLevelStat;
import org.opensearch.ml.stats.MLStat;
import org.opensearch.ml.stats.MLStatLevel;
import org.opensearch.ml.stats.MLStats;
import org.opensearch.ml.stats.MLStatsInput;
import org.opensearch.ml.stats.suppliers.CounterSupplier;
import org.opensearch.ml.stats.suppliers.SettableSupplier;
import org.opensearch.test.OpenSearchIntegTestCase;
import org.opensearch.transport.TransportService;

import com.google.common.collect.ImmutableSet;

public class MLStatsNodesTransportActionTests extends OpenSearchIntegTestCase {
    private MLStatsNodesTransportAction action;
    private MLStats mlStats;
    private Map<Enum, MLStat<?>> statsMap;
    private MLClusterLevelStat clusterStatName1;
    private MLNodeLevelStat nodeStatName1;
    private Environment environment;

    @Override
    @Before
    public void setUp() throws Exception {
        super.setUp();

        clusterStatName1 = MLClusterLevelStat.ML_MODEL_COUNT;
        nodeStatName1 = MLNodeLevelStat.ML_NODE_EXECUTING_TASK_COUNT;

        statsMap = new HashMap<>() {
            {
                put(nodeStatName1, new MLStat<>(false, new CounterSupplier()));
                put(clusterStatName1, new MLStat<>(true, new CounterSupplier()));
                put(MLNodeLevelStat.ML_NODE_JVM_HEAP_USAGE, new MLStat<>(true, new SettableSupplier()));
            }
        };

        mlStats = new MLStats(statsMap);
        environment = mock(Environment.class);
        Settings settings = Settings.builder().build();
        when(environment.settings()).thenReturn(settings);

        action = new MLStatsNodesTransportAction(
            client().threadPool(),
            clusterService(),
            mock(TransportService.class),
            mock(ActionFilters.class),
            mlStats,
            environment
        );
    }

    public void testNewNodeRequest() {
        String nodeId = "nodeId1";
        MLStatsNodesRequest mlStatsNodesRequest = new MLStatsNodesRequest(new String[] { nodeId }, new MLStatsInput());

        MLStatsNodeRequest mlStatsNodeRequest1 = new MLStatsNodeRequest(mlStatsNodesRequest);
        MLStatsNodeRequest mlStatsNodeRequest2 = action.newNodeRequest(mlStatsNodesRequest);

        assertEquals(mlStatsNodeRequest1.getMlStatsNodesRequest(), mlStatsNodeRequest2.getMlStatsNodesRequest());
    }

    public void testNewNodeResponse() throws IOException {
        Map<MLNodeLevelStat, Object> statValues = new HashMap<>();
        DiscoveryNode localNode = new DiscoveryNode("node0", buildNewFakeTransportAddress(), Version.CURRENT);
        MLStatsNodeResponse statsNodeResponse = new MLStatsNodeResponse(localNode, statValues);
        BytesStreamOutput out = new BytesStreamOutput();
        statsNodeResponse.writeTo(out);
        StreamInput in = out.bytes().streamInput();
        MLStatsNodeResponse newStatsNodeResponse = action.newNodeResponse(in);
        Assert.assertEquals(statsNodeResponse.getNodeStats().size(), newStatsNodeResponse.getNodeStats().size());
        for (Enum statName : newStatsNodeResponse.getNodeStats().keySet()) {
            Assert.assertTrue(statsNodeResponse.getNodeStats().containsKey(statName));
        }
    }

    public void testNodeOperation() {
        String nodeId = clusterService().localNode().getId();
        MLStatsNodesRequest mlStatsNodesRequest = new MLStatsNodesRequest(new String[] { nodeId }, new MLStatsInput());

        ImmutableSet<MLNodeLevelStat> statsToBeRetrieved = ImmutableSet.of(nodeStatName1);
        mlStatsNodesRequest.addNodeLevelStats(statsToBeRetrieved);

        MLStatsNodeResponse response = action.nodeOperation(new MLStatsNodeRequest(mlStatsNodesRequest));

        Map<MLNodeLevelStat, Object> stats = response.getNodeStats();

        Assert.assertEquals(1, stats.size());
        for (Enum statName : stats.keySet()) {
            Assert.assertTrue(statsToBeRetrieved.contains(statName));
        }
    }

    public void testNodeOperationWithJvmHeapUsage() {
        String nodeId = clusterService().localNode().getId();
        MLStatsNodesRequest mlStatsNodesRequest = new MLStatsNodesRequest(new String[] { nodeId }, new MLStatsInput());

        Set<MLNodeLevelStat> statsToBeRetrieved = ImmutableSet.of(MLNodeLevelStat.ML_NODE_JVM_HEAP_USAGE);

        mlStatsNodesRequest.addNodeLevelStats(statsToBeRetrieved);

        MLStatsNodeResponse response = action.nodeOperation(new MLStatsNodeRequest(mlStatsNodesRequest));

        Map<MLNodeLevelStat, Object> stats = response.getNodeStats();

        Assert.assertEquals(statsToBeRetrieved.size(), stats.size());
        for (Enum statName : stats.keySet()) {
            Assert.assertTrue(statsToBeRetrieved.contains(statName));
        }
    }

    public void testNodeOperation_NoNodeLevelStat() {
        String nodeId = clusterService().localNode().getId();
        MLStatsInput mlStatsInput = MLStatsInput.builder().targetStatLevels(EnumSet.of(MLStatLevel.ALGORITHM)).build();
        MLStatsNodesRequest mlStatsNodesRequest = new MLStatsNodesRequest(new String[] { nodeId }, mlStatsInput);

        MLStatsNodeResponse response = action.nodeOperation(new MLStatsNodeRequest(mlStatsNodesRequest));

        Map<MLNodeLevelStat, Object> stats = response.getNodeStats();
        assertEquals(0, stats.size());
    }

    public void testNodeOperation_NoNodeLevelStat_AlgoStat() {
        MLStats mlStats = new MLStats(statsMap);
        mlStats.createCounterStatIfAbsent(FunctionName.KMEANS, ActionName.TRAIN, MLActionLevelStat.ML_ACTION_REQUEST_COUNT).increment();

        MLStatsNodesTransportAction action = new MLStatsNodesTransportAction(
            client().threadPool(),
            clusterService(),
            mock(TransportService.class),
            mock(ActionFilters.class),
            mlStats,
            environment
        );

        String nodeId = clusterService().localNode().getId();
        MLStatsInput mlStatsInput = MLStatsInput.builder().targetStatLevels(EnumSet.of(MLStatLevel.ALGORITHM)).build();
        MLStatsNodesRequest mlStatsNodesRequest = new MLStatsNodesRequest(new String[] { nodeId }, mlStatsInput);

        MLStatsNodeResponse response = action.nodeOperation(new MLStatsNodeRequest(mlStatsNodesRequest));

        Map<MLNodeLevelStat, Object> stats = response.getNodeStats();
        assertEquals(0, stats.size());
        Map<FunctionName, MLAlgoStats> algorithmStats = response.getAlgorithmStats();
        assertEquals(1, algorithmStats.size());
        assertTrue(algorithmStats.containsKey(FunctionName.KMEANS));
        Map<ActionName, MLActionStats> algoStats = algorithmStats.get(FunctionName.KMEANS).getAlgoStats();
        assertTrue(algoStats.containsKey(ActionName.TRAIN));
        assertEquals(1l, algoStats.get(ActionName.TRAIN).getAlgoActionStat().get(MLActionLevelStat.ML_ACTION_REQUEST_COUNT));
    }

}
