/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.stats;

import static org.opensearch.common.xcontent.ToXContent.EMPTY_PARAMS;
import static org.opensearch.ml.stats.MLActionLevelStat.ML_ACTION_FAILURE_COUNT;
import static org.opensearch.ml.stats.MLActionLevelStat.ML_ACTION_REQUEST_COUNT;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.junit.Before;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.ml.utils.TestHelper;
import org.opensearch.test.OpenSearchTestCase;

import com.google.common.collect.ImmutableSet;

public class MLActionStatsTests extends OpenSearchTestCase {

    private MLActionStats mlActionStats;
    private long requestCount = 200;
    private long failureCount = 100;

    @Before
    public void setup() {
        Map<MLActionLevelStat, Object> algoActionStats = new HashMap<>();
        algoActionStats.put(ML_ACTION_REQUEST_COUNT, requestCount);
        algoActionStats.put(ML_ACTION_FAILURE_COUNT, failureCount);
        mlActionStats = new MLActionStats(algoActionStats);
    }

    public void testSerializationDeserialization() throws IOException {
        BytesStreamOutput output = new BytesStreamOutput();
        mlActionStats.writeTo(output);
        MLActionStats parsedMLActionStats = new MLActionStats(output.bytes().streamInput());
        assertEquals(2, parsedMLActionStats.getAlgoActionStat().size());
        assertEquals(requestCount, parsedMLActionStats.getAlgoActionStat().get(ML_ACTION_REQUEST_COUNT));
        assertEquals(failureCount, parsedMLActionStats.getAlgoActionStat().get(ML_ACTION_FAILURE_COUNT));
    }

    public void testToXContent() throws IOException {
        XContentBuilder builder = XContentBuilder.builder(XContentType.JSON.xContent());
        builder.startObject();
        mlActionStats.toXContent(builder, EMPTY_PARAMS);
        builder.endObject();
        String content = TestHelper.xContentBuilderToString(builder);
        Set<String> validContents = ImmutableSet
            .of(
                "{\"ml_action_request_count\":200,\"ml_action_failure_count\":100}",
                "{\"ml_action_failure_count\":100,\"ml_action_request_count\":200}"
            );
        assertTrue(validContents.contains(content));
    }

    public void testToXContent_Empty() throws IOException {
        Map<MLActionLevelStat, Object> statMap = new HashMap<>();
        MLActionStats stats = new MLActionStats(statMap);
        XContentBuilder builder = XContentBuilder.builder(XContentType.JSON.xContent());
        builder.startObject();
        stats.toXContent(builder, EMPTY_PARAMS);
        builder.endObject();
        String content = TestHelper.xContentBuilderToString(builder);
        assertEquals("{}", content);
    }
}
