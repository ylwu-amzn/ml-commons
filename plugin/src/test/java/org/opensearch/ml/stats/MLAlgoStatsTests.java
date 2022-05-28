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

public class MLAlgoStatsTests extends OpenSearchTestCase {
    private MLAlgoStats mlAlgoStats;
    private MLActionStats mlActionStats;
    private long requestCount = 200;
    private long failureCount = 100;

    @Before
    public void setup() {
        Map<MLActionLevelStat, Object> algoActionStats = new HashMap<>();
        algoActionStats.put(ML_ACTION_REQUEST_COUNT, requestCount);
        algoActionStats.put(ML_ACTION_FAILURE_COUNT, failureCount);
        mlActionStats = new MLActionStats(algoActionStats);

        Map<ActionName, MLActionStats> algoStats = new HashMap<>();
        algoStats.put(ActionName.TRAIN, mlActionStats);
        mlAlgoStats = new MLAlgoStats(algoStats);
    }

    public void testSerializationDeserialization() throws IOException {
        BytesStreamOutput output = new BytesStreamOutput();
        mlAlgoStats.writeTo(output);
        MLAlgoStats parsedMLAlgoStats = new MLAlgoStats(output.bytes().streamInput());
        MLActionStats parsedMLActionStats = parsedMLAlgoStats.getAlgoStats().get(ActionName.TRAIN);
        assertEquals(2, parsedMLActionStats.getAlgoActionStat().size());
        assertEquals(requestCount, parsedMLActionStats.getAlgoActionStat().get(ML_ACTION_REQUEST_COUNT));
        assertEquals(failureCount, parsedMLActionStats.getAlgoActionStat().get(ML_ACTION_FAILURE_COUNT));
    }

    public void testToXContent() throws IOException {
        XContentBuilder builder = XContentBuilder.builder(XContentType.JSON.xContent());
        builder.startObject();
        mlAlgoStats.toXContent(builder, EMPTY_PARAMS);
        builder.endObject();
        String content = TestHelper.xContentBuilderToString(builder);
        Set<String> validContents = ImmutableSet
            .of(
                "{\"train\":{\"ml_action_request_count\":200,\"ml_action_failure_count\":100}}",
                "{\"train\":{\"ml_action_failure_count\":100,\"ml_action_request_count\":200}}"
            );
        assertTrue(validContents.contains(content));
    }

    public void testToXContent_Empty() throws IOException {
        Map<ActionName, MLActionStats> statMap = new HashMap<>();
        MLAlgoStats stats = new MLAlgoStats(statMap);
        XContentBuilder builder = XContentBuilder.builder(XContentType.JSON.xContent());
        builder.startObject();
        stats.toXContent(builder, EMPTY_PARAMS);
        builder.endObject();
        String content = TestHelper.xContentBuilderToString(builder);
        assertEquals("{}", content);
    }
}
