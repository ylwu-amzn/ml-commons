/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.stats;

import static org.opensearch.common.xcontent.ToXContent.EMPTY_PARAMS;

import java.io.IOException;
import java.util.EnumSet;

import org.junit.Before;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.utils.TestHelper;
import org.opensearch.test.OpenSearchTestCase;

import com.google.common.collect.ImmutableSet;

public class MLStatsInputTests extends OpenSearchTestCase {

    private MLStatsInput mlStatsInput;
    private String node1 = "node1";
    private String node2 = "node2";

    @Before
    public void setup() {
        mlStatsInput = MLStatsInput
            .builder()
            .targetStatLevels(EnumSet.allOf(MLStatLevel.class))
            .clusterLevelStats(EnumSet.allOf(MLClusterLevelStat.class))
            .nodeLevelStats(EnumSet.allOf(MLNodeLevelStat.class))
            .actionLevelStats(EnumSet.allOf(MLActionLevelStat.class))
            .nodeIds(ImmutableSet.of(node1, node2))
            .algorithms(EnumSet.allOf(FunctionName.class))
            .actions(EnumSet.allOf(ActionName.class))
            .build();
    }

    public void testSerializationDeserialization() throws IOException {
        BytesStreamOutput output = new BytesStreamOutput();
        mlStatsInput.writeTo(output);
        MLStatsInput parsedMLStatsInput = new MLStatsInput(output.bytes().streamInput());
        verifyParsedMLStatsInput(parsedMLStatsInput);
    }

    public void testParseMLStatsInput() throws IOException {
        XContentBuilder builder = XContentBuilder.builder(XContentType.JSON.xContent());
        mlStatsInput.toXContent(builder, EMPTY_PARAMS);
        String content = TestHelper.xContentBuilderToString(builder);
        XContentParser parser = TestHelper.parser(content);
        MLStatsInput parsedMLStatsInput = MLStatsInput.parse(parser);
        verifyParsedMLStatsInput(parsedMLStatsInput);
    }

    private void verifyParsedMLStatsInput(MLStatsInput parsedMLStatsInput) {
        assertArrayEquals(
            mlStatsInput.getTargetStatLevels().toArray(new MLStatLevel[0]),
            parsedMLStatsInput.getTargetStatLevels().toArray(new MLStatLevel[0])
        );
        assertArrayEquals(
            mlStatsInput.getClusterLevelStats().toArray(new MLClusterLevelStat[0]),
            parsedMLStatsInput.getClusterLevelStats().toArray(new MLClusterLevelStat[0])
        );
        assertArrayEquals(
            mlStatsInput.getNodeLevelStats().toArray(new MLNodeLevelStat[0]),
            parsedMLStatsInput.getNodeLevelStats().toArray(new MLNodeLevelStat[0])
        );
        assertArrayEquals(
            mlStatsInput.getActionLevelStats().toArray(new MLActionLevelStat[0]),
            parsedMLStatsInput.getActionLevelStats().toArray(new MLActionLevelStat[0])
        );
        assertArrayEquals(
            mlStatsInput.getAlgorithms().toArray(new FunctionName[0]),
            parsedMLStatsInput.getAlgorithms().toArray(new FunctionName[0])
        );
        assertArrayEquals(mlStatsInput.getActions().toArray(new ActionName[0]), parsedMLStatsInput.getActions().toArray(new ActionName[0]));
        assertEquals(2, parsedMLStatsInput.getNodeIds().size());
        assertTrue(parsedMLStatsInput.getNodeIds().contains(node1));
        assertTrue(parsedMLStatsInput.getNodeIds().contains(node2));
    }

}
