/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.ml.indices.MLIndicesHandler.ML_MODEL_INDEX;
import static org.opensearch.ml.utils.TestHelper.getStatsRestRequest;

import java.io.IOException;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.rules.ExpectedException;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.Version;
import org.opensearch.action.ActionListener;
import org.opensearch.client.node.NodeClient;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.node.DiscoveryNodeRole;
import org.opensearch.cluster.node.DiscoveryNodes;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.bytes.BytesReference;
import org.opensearch.common.collect.ImmutableOpenMap;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.transport.TransportAddress;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.ml.action.stats.MLStatsNodeResponse;
import org.opensearch.ml.action.stats.MLStatsNodesAction;
import org.opensearch.ml.action.stats.MLStatsNodesRequest;
import org.opensearch.ml.action.stats.MLStatsNodesResponse;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.plugin.MachineLearningPlugin;
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
import org.opensearch.ml.utils.IndexUtils;
import org.opensearch.rest.BytesRestResponse;
import org.opensearch.rest.RestChannel;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.RestStatus;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.test.rest.FakeRestRequest;
import org.opensearch.threadpool.TestThreadPool;
import org.opensearch.threadpool.ThreadPool;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

public class RestStatsMLActionTests extends OpenSearchTestCase {
    @Rule
    public ExpectedException expectedEx = ExpectedException.none();

    RestStatsMLAction restAction;
    MLStats mlStats;
    @Mock
    ClusterService clusterService;
    @Mock
    IndexUtils indexUtils;

    @Mock
    RestChannel channel;
    ThreadPool threadPool;
    NodeClient client;
    DiscoveryNode node;

    String clusterNameStr = "test cluster";
    ClusterName clusterName;
    private static final AtomicInteger portGenerator = new AtomicInteger();
    ClusterState testState;

    long mlModelCount = 10;
    long nodeTotalRequestCount = 100;
    long kmeansTrainRequestCount = 20;

    @Before
    public void setup() throws IOException {
        MockitoAnnotations.openMocks(this);
        Map<Enum, MLStat<?>> statMap = ImmutableMap
            .<Enum, MLStat<?>>builder()
            .put(MLNodeLevelStat.ML_NODE_EXECUTING_TASK_COUNT, new MLStat<>(false, new CounterSupplier()))
            .build();
        mlStats = new MLStats(statMap);
        threadPool = new TestThreadPool(this.getClass().getSimpleName() + "ThreadPool");
        client = spy(new NodeClient(Settings.EMPTY, threadPool));
        restAction = new RestStatsMLAction(mlStats, clusterService, indexUtils);

        testState = setupTestClusterState();
        when(clusterService.state()).thenReturn(testState);

        clusterName = new ClusterName(clusterNameStr);
        doAnswer(invocation -> {
            ActionListener<Long> actionListener = invocation.getArgument(1);
            actionListener.onResponse(mlModelCount);
            return null;
        }).when(indexUtils).getNumberOfDocumentsInIndex(anyString(), any());

        when(channel.newBuilder()).thenReturn(XContentFactory.jsonBuilder());
    }

    @Override
    public void tearDown() throws Exception {
        super.tearDown();
        threadPool.shutdown();
        client.close();
    }

    private ClusterState setupTestClusterState() {
        Set<DiscoveryNodeRole> roleSet = new HashSet<>();
        roleSet.add(DiscoveryNodeRole.DATA_ROLE);
        node = new DiscoveryNode(
            "node",
            new TransportAddress(TransportAddress.META_ADDRESS, portGenerator.incrementAndGet()),
            new HashMap<>(),
            roleSet,
            Version.CURRENT
        );
        Metadata metadata = new Metadata.Builder()
            .indices(
                ImmutableOpenMap
                    .<String, IndexMetadata>builder()
                    .fPut(
                        ML_MODEL_INDEX,
                        IndexMetadata
                            .builder("test")
                            .settings(
                                Settings
                                    .builder()
                                    .put("index.number_of_shards", 1)
                                    .put("index.number_of_replicas", 1)
                                    .put("index.version.created", Version.CURRENT.id)
                            )
                            .build()
                    )
                    .build()
            )
            .build();
        return new ClusterState(
            clusterName,
            123l,
            "111111",
            metadata,
            null,
            DiscoveryNodes.builder().add(node).build(),
            null,
            null,
            0,
            false
        );
    }

    public void testPrepareRequest_AllStateLevels() throws Exception {
        MLStatsInput mlStatsInput = MLStatsInput.builder().targetStatLevels(EnumSet.allOf(MLStatLevel.class)).build();
        RestRequest request = getStatsRestRequest(mlStatsInput);
        restAction.handleRequest(request, channel, client);

        ArgumentCaptor<MLStatsNodesRequest> argumentCaptor = ArgumentCaptor.forClass(MLStatsNodesRequest.class);
        verify(client, times(1)).execute(eq(MLStatsNodesAction.INSTANCE), argumentCaptor.capture(), any());
        MLStatsInput input = argumentCaptor.getValue().getMlStatsInput();
        assertEquals(mlStatsInput.getTargetStatLevels().size(), input.getTargetStatLevels().size());
        for (MLStatLevel statLevel : mlStatsInput.getTargetStatLevels()) {
            assertTrue(input.getTargetStatLevels().contains(statLevel));
        }
    }

    public void testPrepareRequest_ClusterLevelStates() throws Exception {
        MLStatsInput mlStatsInput = MLStatsInput.builder().targetStatLevels(EnumSet.of(MLStatLevel.CLUSTER)).build();
        RestRequest request = getStatsRestRequest(mlStatsInput);
        restAction.handleRequest(request, channel, client);

        ArgumentCaptor<BytesRestResponse> argumentCaptor = ArgumentCaptor.forClass(BytesRestResponse.class);
        verify(channel, times(1)).sendResponse(argumentCaptor.capture());
        BytesRestResponse restResponse = argumentCaptor.getValue();
        assertEquals(RestStatus.OK, restResponse.status());
        BytesReference content = restResponse.content();
        assertEquals("{\"ml_model_count\":10}", content.utf8ToString());
    }

    public void testPrepareRequest_ClusterAndNodeLevelStates() throws Exception {
        prepareResponse();

        MLStatsInput mlStatsInput = MLStatsInput.builder().targetStatLevels(EnumSet.of(MLStatLevel.CLUSTER, MLStatLevel.NODE)).build();
        RestRequest request = getStatsRestRequest(mlStatsInput);
        restAction.handleRequest(request, channel, client);

        ArgumentCaptor<MLStatsNodesRequest> inputArgumentCaptor = ArgumentCaptor.forClass(MLStatsNodesRequest.class);
        verify(client, times(1)).execute(eq(MLStatsNodesAction.INSTANCE), inputArgumentCaptor.capture(), any());
        MLStatsInput input = inputArgumentCaptor.getValue().getMlStatsInput();
        assertEquals(mlStatsInput.getTargetStatLevels().size(), input.getTargetStatLevels().size());
        for (MLStatLevel statLevel : mlStatsInput.getTargetStatLevels()) {
            assertTrue(input.getTargetStatLevels().contains(statLevel));
        }

        ArgumentCaptor<BytesRestResponse> argumentCaptor = ArgumentCaptor.forClass(BytesRestResponse.class);
        verify(channel, times(1)).sendResponse(argumentCaptor.capture());
        BytesRestResponse restResponse = argumentCaptor.getValue();
        assertEquals(RestStatus.OK, restResponse.status());
        BytesReference content = restResponse.content();
        assertEquals(
            "{\"ml_model_count\":10,\"nodes\":{\"node\":{\"ml_node_total_request_count\":100,\"algorithms\":{\"kmeans\":{\"train\":{\"ml_action_request_count\":20}}}}}}",
            content.utf8ToString()
        );
    }

    private void prepareResponse() {
        doAnswer(invocation -> {
            ActionListener<MLStatsNodesResponse> actionListener = invocation.getArgument(2);
            List<MLStatsNodeResponse> nodes = new ArrayList<>();
            Map<MLNodeLevelStat, Object> nodeStats = ImmutableMap.of(MLNodeLevelStat.ML_NODE_TOTAL_REQUEST_COUNT, nodeTotalRequestCount);
            Map<FunctionName, MLAlgoStats> algoStats = new HashMap<>();
            Map<ActionName, MLActionStats> actionStats = ImmutableMap
                .of(
                    ActionName.TRAIN,
                    new MLActionStats(ImmutableMap.of(MLActionLevelStat.ML_ACTION_REQUEST_COUNT, kmeansTrainRequestCount))
                );
            algoStats.put(FunctionName.KMEANS, new MLAlgoStats(actionStats));
            MLStatsNodeResponse nodeResponse = new MLStatsNodeResponse(node, nodeStats, algoStats);
            nodes.add(nodeResponse);
            MLStatsNodesResponse statsResponse = new MLStatsNodesResponse(clusterName, nodes, ImmutableList.of());
            actionListener.onResponse(statsResponse);
            return null;
        }).when(client).execute(eq(MLStatsNodesAction.INSTANCE), any(), any());
    }

    public void testPrepareRequest_ClusterAndNodeLevelStates_Failure() throws Exception {
        doAnswer(invocation -> {
            ActionListener<MLStatsNodesResponse> actionListener = invocation.getArgument(2);
            actionListener.onFailure(new RuntimeException("test failure"));
            return null;
        }).when(client).execute(eq(MLStatsNodesAction.INSTANCE), any(), any());

        MLStatsInput mlStatsInput = MLStatsInput.builder().targetStatLevels(EnumSet.of(MLStatLevel.CLUSTER, MLStatLevel.NODE)).build();
        RestRequest request = getStatsRestRequest(mlStatsInput);
        restAction.handleRequest(request, channel, client);

        ArgumentCaptor<MLStatsNodesRequest> inputArgumentCaptor = ArgumentCaptor.forClass(MLStatsNodesRequest.class);
        verify(client, times(1)).execute(eq(MLStatsNodesAction.INSTANCE), inputArgumentCaptor.capture(), any());
        MLStatsInput input = inputArgumentCaptor.getValue().getMlStatsInput();
        assertEquals(mlStatsInput.getTargetStatLevels().size(), input.getTargetStatLevels().size());
        for (MLStatLevel statLevel : mlStatsInput.getTargetStatLevels()) {
            assertTrue(input.getTargetStatLevels().contains(statLevel));
        }

        ArgumentCaptor<BytesRestResponse> argumentCaptor = ArgumentCaptor.forClass(BytesRestResponse.class);
        verify(channel, times(1)).sendResponse(argumentCaptor.capture());
        BytesRestResponse restResponse = argumentCaptor.getValue();
        assertEquals(RestStatus.INTERNAL_SERVER_ERROR, restResponse.status());
        BytesReference content = restResponse.content();
        // Error happened when generate failure response, then will return general error message
        assertEquals("Failed to get ML node level stats", content.utf8ToString());

        when(channel.request()).thenReturn(request);
        when(channel.newErrorBuilder()).thenReturn(XContentFactory.jsonBuilder());
        when(channel.detailedErrorsEnabled()).thenReturn(false);
        restAction.handleRequest(request, channel, client);
        argumentCaptor = ArgumentCaptor.forClass(BytesRestResponse.class);
        verify(channel, times(2)).sendResponse(argumentCaptor.capture());
        restResponse = argumentCaptor.getValue();
        assertEquals(RestStatus.INTERNAL_SERVER_ERROR, restResponse.status());
        content = restResponse.content();
        // Return exception directly in normal case
        assertEquals("{\"error\":\"No OpenSearchException found\",\"status\":500}", content.utf8ToString());
    }

    public void testPrepareRequest_ClusterAndNodeLevelStates_NoRequestContent() throws Exception {
        prepareResponse();

        RestRequest request = getStatsRestRequest();
        restAction.handleRequest(request, channel, client);

        ArgumentCaptor<MLStatsNodesRequest> inputArgumentCaptor = ArgumentCaptor.forClass(MLStatsNodesRequest.class);
        verify(client, times(1)).execute(eq(MLStatsNodesAction.INSTANCE), inputArgumentCaptor.capture(), any());
        MLStatsInput input = inputArgumentCaptor.getValue().getMlStatsInput();
        assertEquals(MLStatLevel.values().length, input.getTargetStatLevels().size());
        for (MLStatLevel statLevel : MLStatLevel.values()) {
            assertTrue(input.getTargetStatLevels().contains(statLevel));
        }

        ArgumentCaptor<BytesRestResponse> argumentCaptor = ArgumentCaptor.forClass(BytesRestResponse.class);
        verify(channel, times(1)).sendResponse(argumentCaptor.capture());
        BytesRestResponse restResponse = argumentCaptor.getValue();
        assertEquals(RestStatus.OK, restResponse.status());
        BytesReference content = restResponse.content();
        assertEquals(
            "{\"ml_model_count\":10,\"nodes\":{\"node\":{\"ml_node_total_request_count\":100,\"algorithms\":{\"kmeans\":{\"train\":{\"ml_action_request_count\":20}}}}}}",
            content.utf8ToString()
        );
    }

    public void testPrepareRequest_ClusterAndNodeLevelStates_RequestParams() throws Exception {
        prepareResponse();

        RestRequest request = getStatsRestRequest(
            node.getId(),
            MLClusterLevelStat.ML_MODEL_COUNT + "," + MLNodeLevelStat.ML_NODE_TOTAL_MODEL_COUNT
        );
        restAction.handleRequest(request, channel, client);

        ArgumentCaptor<MLStatsNodesRequest> inputArgumentCaptor = ArgumentCaptor.forClass(MLStatsNodesRequest.class);
        verify(client, times(1)).execute(eq(MLStatsNodesAction.INSTANCE), inputArgumentCaptor.capture(), any());
        MLStatsInput input = inputArgumentCaptor.getValue().getMlStatsInput();
        assertEquals(2, input.getTargetStatLevels().size());
        assertTrue(input.getTargetStatLevels().contains(MLStatLevel.CLUSTER));
        assertTrue(input.getTargetStatLevels().contains(MLStatLevel.NODE));
        assertEquals(1, input.getClusterLevelStats().size());
        assertTrue(input.getClusterLevelStats().contains(MLClusterLevelStat.ML_MODEL_COUNT));
        assertTrue(input.getNodeLevelStats().contains(MLNodeLevelStat.ML_NODE_TOTAL_MODEL_COUNT));

        ArgumentCaptor<BytesRestResponse> argumentCaptor = ArgumentCaptor.forClass(BytesRestResponse.class);
        verify(channel, times(1)).sendResponse(argumentCaptor.capture());
        BytesRestResponse restResponse = argumentCaptor.getValue();
        assertEquals(RestStatus.OK, restResponse.status());
        BytesReference content = restResponse.content();
        assertEquals(
            "{\"ml_model_count\":10,\"nodes\":{\"node\":{\"ml_node_total_request_count\":100,\"algorithms\":{\"kmeans\":{\"train\":{\"ml_action_request_count\":20}}}}}}",
            content.utf8ToString()
        );
    }

    public void testPrepareRequest_ClusterAndNodeLevelStates_RequestParams_NodeLevelStat() throws Exception {
        prepareResponse();

        RestRequest request = getStatsRestRequest(node.getId(), MLNodeLevelStat.ML_NODE_TOTAL_MODEL_COUNT.name());
        restAction.handleRequest(request, channel, client);

        ArgumentCaptor<MLStatsNodesRequest> inputArgumentCaptor = ArgumentCaptor.forClass(MLStatsNodesRequest.class);
        verify(client, times(1)).execute(eq(MLStatsNodesAction.INSTANCE), inputArgumentCaptor.capture(), any());
        MLStatsInput input = inputArgumentCaptor.getValue().getMlStatsInput();
        assertEquals(1, input.getTargetStatLevels().size());
        assertTrue(input.getTargetStatLevels().contains(MLStatLevel.NODE));
        assertEquals(0, input.getClusterLevelStats().size());
        assertEquals(1, input.getNodeLevelStats().size());
        assertTrue(input.getNodeLevelStats().contains(MLNodeLevelStat.ML_NODE_TOTAL_MODEL_COUNT));

        ArgumentCaptor<BytesRestResponse> argumentCaptor = ArgumentCaptor.forClass(BytesRestResponse.class);
        verify(channel, times(1)).sendResponse(argumentCaptor.capture());
        BytesRestResponse restResponse = argumentCaptor.getValue();
        assertEquals(RestStatus.OK, restResponse.status());
        BytesReference content = restResponse.content();
        assertEquals(
            "{\"nodes\":{\"node\":{\"ml_node_total_request_count\":100,\"algorithms\":{\"kmeans\":{\"train\":{\"ml_action_request_count\":20}}}}}}",
            content.utf8ToString()
        );
    }

    public void testCreateMlStatsInputFromRequestParams_NodeStat() {
        RestRequest request = getStatsRestRequest(node.getId(), MLNodeLevelStat.ML_NODE_TOTAL_MODEL_COUNT.name().toLowerCase(Locale.ROOT));
        MLStatsInput input = restAction.createMlStatsInputFromRequestParams(request);
        assertEquals(1, input.getTargetStatLevels().size());
        assertTrue(input.getTargetStatLevels().contains(MLStatLevel.NODE));
        assertTrue(input.getNodeLevelStats().contains(MLNodeLevelStat.ML_NODE_TOTAL_MODEL_COUNT));
        assertEquals(0, input.getClusterLevelStats().size());
    }

    public void testCreateMlStatsInputFromRequestParams_ClusterStat() {
        RestRequest request = getStatsRestRequest(node.getId(), MLClusterLevelStat.ML_MODEL_COUNT.name().toLowerCase(Locale.ROOT));
        MLStatsInput input = restAction.createMlStatsInputFromRequestParams(request);
        assertEquals(1, input.getTargetStatLevels().size());
        assertTrue(input.getTargetStatLevels().contains(MLStatLevel.CLUSTER));
        assertTrue(input.getClusterLevelStats().contains(MLClusterLevelStat.ML_MODEL_COUNT));
        assertEquals(0, input.getNodeLevelStats().size());
    }

    public void testSplitCommaSeparatedParam() {
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
