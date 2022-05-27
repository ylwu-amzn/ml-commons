/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.stats;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
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
import org.opensearch.ml.stats.InternalStatNames;
import org.opensearch.ml.stats.MLStat;
import org.opensearch.ml.stats.MLStats;
import org.opensearch.ml.stats.suppliers.CounterSupplier;
import org.opensearch.ml.stats.suppliers.SettableSupplier;
import org.opensearch.test.OpenSearchIntegTestCase;
import org.opensearch.transport.TransportService;

public class MLStatsNodesTransportActionTests extends OpenSearchIntegTestCase {
    private MLStatsNodesTransportAction action;
    private MLStats mlStats;
    private Map<String, MLStat<?>> statsMap;
    private String clusterStatName1;
    private String nodeStatName1;

    @Override
    @Before
    public void setUp() throws Exception {
        super.setUp();

        clusterStatName1 = "clusterStat1";
        nodeStatName1 = "nodeStat1";

        statsMap = new HashMap<String, MLStat<?>>() {
            {
                put(nodeStatName1, new MLStat<>(false, new CounterSupplier()));
                put(clusterStatName1, new MLStat<>(true, new CounterSupplier()));
                put(InternalStatNames.JVM_HEAP_USAGE.getName(), new MLStat<>(true, new SettableSupplier()));
            }
        };

        mlStats = new MLStats(statsMap);
        Environment environment = mock(Environment.class);
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
        MLStatsNodesRequest mlStatsNodesRequest = new MLStatsNodesRequest(nodeId);

        MLStatsNodeRequest mlStatsNodeRequest1 = new MLStatsNodeRequest(mlStatsNodesRequest);
        MLStatsNodeRequest mlStatsNodeRequest2 = action.newNodeRequest(mlStatsNodesRequest);

        assertEquals(mlStatsNodeRequest1.getMlStatsNodesRequest(), mlStatsNodeRequest2.getMlStatsNodesRequest());
    }

    public void testNewNodeResponse() throws IOException {
        Map<String, Object> statValues = new HashMap<>();
        DiscoveryNode localNode = new DiscoveryNode("node0", buildNewFakeTransportAddress(), Version.CURRENT);
        MLStatsNodeResponse statsNodeResponse = new MLStatsNodeResponse(localNode, statValues);
        BytesStreamOutput out = new BytesStreamOutput();
        statsNodeResponse.writeTo(out);
        StreamInput in = out.bytes().streamInput();
        MLStatsNodeResponse newStatsNodeResponse = action.newNodeResponse(in);
        Assert.assertEquals(statsNodeResponse.getNodeStats().size(), newStatsNodeResponse.getNodeStats().size());
        for (String statName : newStatsNodeResponse.getNodeStats().keySet()) {
            Assert.assertTrue(statsNodeResponse.getNodeStats().containsKey(statName));
        }
    }

    public void testNodeOperation() {
        String nodeId = clusterService().localNode().getId();
        MLStatsNodesRequest mlStatsNodesRequest = new MLStatsNodesRequest((nodeId));
        mlStatsNodesRequest.clear();

        Set<String> statsToBeRetrieved = new HashSet<>(Arrays.asList(nodeStatName1));

        for (String stat : statsToBeRetrieved) {
            mlStatsNodesRequest.addStat(stat);
        }

        MLStatsNodeResponse response = action.nodeOperation(new MLStatsNodeRequest(mlStatsNodesRequest));

        Map<String, Object> stats = response.getNodeStats();

        Assert.assertEquals(statsToBeRetrieved.size(), stats.size());
        for (String statName : stats.keySet()) {
            Assert.assertTrue(statsToBeRetrieved.contains(statName));
        }
    }

    public void testNodeOperationWithJvmHeapUsage() {
        String nodeId = clusterService().localNode().getId();
        MLStatsNodesRequest mlStatsNodesRequest = new MLStatsNodesRequest((nodeId));
        mlStatsNodesRequest.clear();

        Set<String> statsToBeRetrieved = new HashSet<>(Arrays.asList(nodeStatName1, InternalStatNames.JVM_HEAP_USAGE.getName()));

        for (String stat : statsToBeRetrieved) {
            mlStatsNodesRequest.addStat(stat);
        }

        MLStatsNodeResponse response = action.nodeOperation(new MLStatsNodeRequest(mlStatsNodesRequest));

        Map<String, Object> stats = response.getNodeStats();

        Assert.assertEquals(statsToBeRetrieved.size(), stats.size());
        for (String statName : stats.keySet()) {
            Assert.assertTrue(statsToBeRetrieved.contains(statName));
        }
    }

    public void testNodeOperationNotSupportedStat() {
        String nodeId = clusterService().localNode().getId();
        MLStatsNodesRequest mlStatsNodesRequest = new MLStatsNodesRequest((nodeId));
        mlStatsNodesRequest.clear();

        Set<String> statsToBeRetrieved = new HashSet<>(Arrays.asList("notSupportedStat"));

        for (String stat : statsToBeRetrieved) {
            mlStatsNodesRequest.addStat(stat);
        }

        MLStatsNodeResponse response = action.nodeOperation(new MLStatsNodeRequest(mlStatsNodesRequest));

        Map<String, Object> stats = response.getNodeStats();

        Assert.assertEquals(0, stats.size());
    }

}
