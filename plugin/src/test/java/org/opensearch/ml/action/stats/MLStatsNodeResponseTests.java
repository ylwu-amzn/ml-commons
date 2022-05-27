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
import org.opensearch.Version;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.ml.stats.StatNames;
import org.opensearch.ml.utils.TestHelper;
import org.opensearch.test.OpenSearchTestCase;

public class MLStatsNodeResponseTests extends OpenSearchTestCase {
    private MLStatsNodeResponse response;

    @Before
    public void setup() {
        DiscoveryNode node = new DiscoveryNode("node0", buildNewFakeTransportAddress(), Version.CURRENT);
        Map<String, Object> statsToValues = new HashMap<>();
        statsToValues.put(StatNames.ML_NODE_TOTAL_REQUEST_COUNT, 100);
        response = new MLStatsNodeResponse(node, statsToValues);
    }

    public void testSerializationDeserialization() throws IOException {
        DiscoveryNode localNode = new DiscoveryNode("node0", buildNewFakeTransportAddress(), Version.CURRENT);
        Map<String, Object> statsToValues = new HashMap<>();
        statsToValues.put("stat1", "value1");
        MLStatsNodeResponse response = new MLStatsNodeResponse(localNode, statsToValues);
        BytesStreamOutput output = new BytesStreamOutput();
        response.writeTo(output);
        MLStatsNodeResponse newResponse = new MLStatsNodeResponse(output.bytes().streamInput());
        Assert.assertEquals(newResponse.getNodeStats().size(), response.getNodeStats().size());
        for (String stat : newResponse.getNodeStats().keySet()) {
            Assert.assertEquals(response.getNodeStats().get(stat), newResponse.getNodeStats().get(stat));
        }
    }

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
        Map<String, Object> statsToValues = new HashMap<>();
        statsToValues.put(StatNames.ML_NODE_TOTAL_REQUEST_COUNT, 100);
        Map<String, MLAlgoActionStats> algoStats = new HashMap<>();
        Map<String, MLAlgoStats> algoActionStats = new HashMap<>();
        Map<String, Object> algoActionStatMap = new HashMap<>();
        algoActionStatMap.put("request_count", 111);
        algoActionStatMap.put("request_failure", 22);
        algoActionStats.put("train", new MLAlgoStats(algoActionStatMap));
        algoStats.put("kmeans", new MLAlgoActionStats(algoActionStats));
        response = new MLStatsNodeResponse(node, statsToValues, algoStats);
        response.toXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.endObject();
        String taskContent = TestHelper.xContentBuilderToString(builder);
        assertEquals(
            "{\"ml_total_request_count\":100,\"algorithms\":{\"kmeans\":{\"train\":{\"request_failure\":22,\"request_count\":111}}}}",
            taskContent
        );
    }

    public void testReadStats() throws IOException {
        BytesStreamOutput output = new BytesStreamOutput();
        response.writeTo(output);
        MLStatsNodeResponse mlStatsNodeResponse = MLStatsNodeResponse.readStats(output.bytes().streamInput());
        Integer expectedValue = (Integer) response.getNodeStats().get(StatNames.ML_NODE_TOTAL_REQUEST_COUNT);
        assertEquals(expectedValue, mlStatsNodeResponse.getNodeStats().get(StatNames.ML_NODE_TOTAL_REQUEST_COUNT));
    }
}
