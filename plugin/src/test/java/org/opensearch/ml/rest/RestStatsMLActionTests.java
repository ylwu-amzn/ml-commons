/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import java.util.Map;
import java.util.Optional;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.rules.ExpectedException;
import org.mockito.Mock;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.ml.plugin.MachineLearningPlugin;
import org.opensearch.ml.stats.MLNodeLevelStat;
import org.opensearch.ml.stats.MLStat;
import org.opensearch.ml.stats.MLStats;
import org.opensearch.ml.stats.suppliers.CounterSupplier;
import org.opensearch.ml.utils.IndexUtils;
import org.opensearch.rest.RestRequest;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.test.rest.FakeRestRequest;

import com.google.common.collect.ImmutableMap;

public class RestStatsMLActionTests extends OpenSearchTestCase {
    @Rule
    public ExpectedException thrown = ExpectedException.none();

    RestStatsMLAction restAction;
    MLStats mlStats;
    @Mock
    ClusterService clusterService;
    @Mock
    IndexUtils indexUtils;

    @Before
    public void setup() {
        Map<Enum, MLStat<?>> statMap = ImmutableMap
            .<Enum, MLStat<?>>builder()
            .put(MLNodeLevelStat.ML_NODE_EXECUTING_TASK_COUNT, new MLStat<>(false, new CounterSupplier()))
            .build();
        mlStats = new MLStats(statMap);
        restAction = new RestStatsMLAction(mlStats, clusterService, indexUtils);
    }

    public void testsplitCommaSeparatedParam() {
        Map<String, String> param = ImmutableMap.<String, String>builder().put("nodeId", "111,222").build();
        FakeRestRequest fakeRestRequest = new FakeRestRequest.Builder(xContentRegistry())
            .withMethod(RestRequest.Method.GET)
            .withPath(MachineLearningPlugin.ML_BASE_URI + "/{nodeId}/stats/")
            .withParams(param)
            .build();
        Optional<String[]> nodeId = restAction.splitCommaSeparatedParam(fakeRestRequest, "nodeId");
        String[] array = nodeId.get();
        Assert.assertEquals(array[0], "111");
        Assert.assertEquals(array[1], "222");
    }

    // public void testIsAllStatsRequested() {
    // List<String> requestedStats1 = new ArrayList<>(Arrays.asList("stat1", "stat2"));
    // Assert.assertTrue(!restAction.isAllStatsRequested(requestedStats1));
    // List<String> requestedStats2 = new ArrayList<>();
    // Assert.assertTrue(restAction.isAllStatsRequested(requestedStats2));
    // List<String> requestedStats3 = new ArrayList<>(Arrays.asList(MLStatsNodesRequest.ALL_STATS_KEY));
    // Assert.assertTrue(restAction.isAllStatsRequested(requestedStats3));
    // }

    // public void testStatsSetContainsAllStatsKey() {
    // thrown.expect(IllegalArgumentException.class);
    // thrown.expectMessage(MLStatsNodesRequest.ALL_STATS_KEY);
    // FakeRestRequest fakeRestRequest = new FakeRestRequest.Builder(xContentRegistry())
    // .withMethod(RestRequest.Method.GET)
    // .withPath(MachineLearningPlugin.ML_BASE_URI + "/{nodeId}/stats/")
    // .build();
    // Set<String> validStats = new HashSet<>();
    // validStats.add("stat1");
    // validStats.add("stat2");
    // List<String> requestedStats = new ArrayList<>(Arrays.asList("stat1", "stat2", MLStatsNodesRequest.ALL_STATS_KEY));
    // restAction.getStatsToBeRetrieved(fakeRestRequest, validStats, requestedStats);
    // }

    // public void testStatsSetContainsInvalidStats() {
    // thrown.expect(IllegalArgumentException.class);
    // thrown.expectMessage("unrecognized");
    // FakeRestRequest fakeRestRequest = new FakeRestRequest.Builder(xContentRegistry())
    // .withMethod(RestRequest.Method.GET)
    // .withPath(MachineLearningPlugin.ML_BASE_URI + "/{nodeId}/stats/")
    // .build();
    // Set<String> validStats = new HashSet<>();
    // validStats.add("stat1");
    // validStats.add("stat2");
    // List<String> requestedStats = new ArrayList<>(Arrays.asList("stat1", "stat2", "invalidStat"));
    // restAction.getStatsToBeRetrieved(fakeRestRequest, validStats, requestedStats);
    // }

    // public void testGetRequestAllStats() {
    // Map<String, String> param = ImmutableMap
    // .<String, String>builder()
    // .put("nodeId", "111,222")
    // .put("stat", MLStatsNodesRequest.ALL_STATS_KEY)
    // .build();
    // FakeRestRequest fakeRestRequest = new FakeRestRequest.Builder(xContentRegistry())
    // .withMethod(RestRequest.Method.GET)
    // .withPath(MachineLearningPlugin.ML_BASE_URI + "/{nodeId}/stats/{stat}")
    // .withParams(param)
    // .build();
    // MLStatsNodesRequest request = restAction.getRequest(fakeRestRequest);
    // Assert.assertEquals(0, request.getStatsToBeRetrieved().size());
    // Assert.assertTrue(request.isRetrieveAllStats());
    // }

    // public void testGetRequestEmptyStats() {
    // Map<String, String> param = ImmutableMap.<String, String>builder().put("nodeId", "111,222").build();
    // FakeRestRequest fakeRestRequest = new FakeRestRequest.Builder(xContentRegistry())
    // .withMethod(RestRequest.Method.GET)
    // .withPath(MachineLearningPlugin.ML_BASE_URI + "/{nodeId}/stats/")
    // .withParams(param)
    // .build();
    // MLStatsNodesRequest request = restAction.getRequest(fakeRestRequest);
    // Assert.assertEquals(0, request.getStatsToBeRetrieved().size());
    // Assert.assertTrue(request.isRetrieveAllStats());
    // }

    // public void testGetRequestSpecifyStats() {
    // Map<String, String> param = ImmutableMap
    // .<String, String>builder()
    // .put("nodeId", "111,222")
    // .put("stat", StatNames.ML_NODE_EXECUTING_TASK_COUNT)
    // .build();
    // FakeRestRequest fakeRestRequest = new FakeRestRequest.Builder(xContentRegistry())
    // .withMethod(RestRequest.Method.GET)
    // .withPath(MachineLearningPlugin.ML_BASE_URI + "/{nodeId}/stats/{stat}")
    // .withParams(param)
    // .build();
    // MLStatsNodesRequest request = restAction.getRequest(fakeRestRequest);
    // Assert.assertEquals(request.getStatsToBeRetrieved().size(), 1);
    // Assert.assertTrue(request.getStatsToBeRetrieved().contains(StatNames.ML_NODE_EXECUTING_TASK_COUNT));
    // }
}
