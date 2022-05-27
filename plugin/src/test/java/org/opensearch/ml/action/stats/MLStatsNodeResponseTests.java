/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.stats;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Ignore;
import org.opensearch.Version;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.stats.ActionName;
import org.opensearch.ml.stats.MLActionLevelStat;
import org.opensearch.ml.stats.MLActionStats;
import org.opensearch.ml.stats.MLAlgoActionStats;
import org.opensearch.ml.stats.MLNodeLevelStat;
import org.opensearch.ml.utils.TestHelper;
import org.opensearch.test.OpenSearchTestCase;

public class MLStatsNodeResponseTests extends OpenSearchTestCase {
    private MLStatsNodeResponse response;

    @Before
    public void setup() {
        DiscoveryNode node = new DiscoveryNode("node0", buildNewFakeTransportAddress(), Version.CURRENT);
        Map<MLNodeLevelStat, Object> statsToValues = new HashMap<>();
        statsToValues.put(MLNodeLevelStat.ML_NODE_TOTAL_REQUEST_COUNT, 100);
        response = new MLStatsNodeResponse(node, statsToValues);
    }

    @Ignore
    public void testSerializationDeserialization() throws IOException {
        DiscoveryNode localNode = new DiscoveryNode("node0", buildNewFakeTransportAddress(), Version.CURRENT);
        Map<MLNodeLevelStat, Object> statsToValues = new HashMap<>();
        statsToValues.put(MLNodeLevelStat.ML_NODE_TOTAL_REQUEST_COUNT, 10l);
        MLStatsNodeResponse response = new MLStatsNodeResponse(localNode, statsToValues);
        BytesStreamOutput output = new BytesStreamOutput();
        response.writeTo(output);
        MLStatsNodeResponse newResponse = new MLStatsNodeResponse(output.bytes().streamInput());
        Assert.assertEquals(newResponse.getNodeStats().size(), response.getNodeStats().size());
        for (MLNodeLevelStat stat : newResponse.getNodeStats().keySet()) {
            Assert.assertEquals(response.getNodeStats().get(stat), newResponse.getNodeStats().get(stat));
        }
    }

    @Ignore
    public void testToXContent() throws IOException {
        XContentBuilder builder = XContentBuilder.builder(XContentType.JSON.xContent());
        builder.startObject();
        response.toXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.endObject();
        String taskContent = TestHelper.xContentBuilderToString(builder);
        assertEquals("{\"ml_total_request_count\":100}", taskContent);
    }

    public void testToXContent_WithAlgoStats() throws IOException {
        XContentBuilder builder = XContentBuilder.builder(XContentType.JSON.xContent());
        builder.startObject();
        DiscoveryNode node = new DiscoveryNode("node0", buildNewFakeTransportAddress(), Version.CURRENT);
        Map<MLNodeLevelStat, Object> statsToValues = new HashMap<>();
        statsToValues.put(MLNodeLevelStat.ML_NODE_TOTAL_REQUEST_COUNT, 100);
        Map<FunctionName, MLAlgoActionStats> algoStats = new HashMap<>();
        Map<ActionName, MLActionStats> algoActionStats = new HashMap<>();
        Map<MLActionLevelStat, Object> algoActionStatMap = new HashMap<>();
        algoActionStatMap.put(MLActionLevelStat.ML_ACTION_REQUEST_COUNT, 111);
        algoActionStatMap.put(MLActionLevelStat.ML_ACTION_FAILURE_COUNT, 22);
        algoActionStats.put(ActionName.TRAIN, new MLActionStats(algoActionStatMap));
        algoStats.put(FunctionName.KMEANS, new MLAlgoActionStats(algoActionStats));
        response = new MLStatsNodeResponse(node, statsToValues, algoStats);
        response.toXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.endObject();
        String taskContent = TestHelper.xContentBuilderToString(builder);
        assertEquals(
            "{\"ml_node_total_request_count\":100,\"algorithms\":{\"kmeans\":{\"train\":{\"ml_action_request_count\":111,\"ml_action_failure_count\":22}}}}",
            taskContent
        );
    }

    @Ignore
    public void testReadStats() throws IOException {
        BytesStreamOutput output = new BytesStreamOutput();
        response.writeTo(output);
        MLStatsNodeResponse mlStatsNodeResponse = MLStatsNodeResponse.readStats(output.bytes().streamInput());
        Integer expectedValue = (Integer) response.getNodeStats().get(MLNodeLevelStat.ML_NODE_TOTAL_REQUEST_COUNT);
        assertEquals(expectedValue, mlStatsNodeResponse.getNodeStats().get(MLNodeLevelStat.ML_NODE_TOTAL_REQUEST_COUNT));
    }
}
