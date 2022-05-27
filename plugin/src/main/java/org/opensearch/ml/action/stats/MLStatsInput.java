/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.stats;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;

import java.io.IOException;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.function.Function;

import lombok.Builder;
import lombok.Getter;

import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;

@Getter
public class MLStatsInput implements Writeable {
    public static final String TARGET_STAT_LEVEL = "target_stat_levels";
    public static final String CLUSTER_LEVEL_STATS = "cluster_level_stats";
    public static final String NODE_LEVEL_STATS = "node_level_stats";
    public static final String NODE_IDS = "node_ids";
    public static final String ALGORITHMS = "algorithms";

    private EnumSet<MLStatLevel> targetStatLevels;
    private EnumSet<MLClusterLevelStat> clusterLevelStats;
    private EnumSet<MLNodeLevelStat> nodeLevelStats;
    private EnumSet<FunctionName> algorithms;
    private Set<String> nodeIds;

    @Builder
    public MLStatsInput(
        EnumSet<MLStatLevel> targetStatLevels,
        EnumSet<MLClusterLevelStat> clusterLevelStats,
        EnumSet<MLNodeLevelStat> nodeLevelStats,
        EnumSet<FunctionName> algorithms,
        Set<String> nodeIds
    ) {
        this.targetStatLevels = targetStatLevels;
        this.clusterLevelStats = clusterLevelStats;
        this.nodeLevelStats = nodeLevelStats;
        this.algorithms = algorithms;
        this.nodeIds = nodeIds;
    }

    public MLStatsInput() {
        this.targetStatLevels = EnumSet.noneOf(MLStatLevel.class);
        this.clusterLevelStats = EnumSet.noneOf(MLClusterLevelStat.class);
        this.nodeLevelStats = EnumSet.noneOf(MLNodeLevelStat.class);
        this.algorithms = EnumSet.noneOf(FunctionName.class);
        this.nodeIds = new HashSet<>();
    }

    public MLStatsInput(StreamInput input) throws IOException {
        if (input.readBoolean()) {
            targetStatLevels = input.readEnumSet(MLStatLevel.class);
        }
        if (input.readBoolean()) {
            clusterLevelStats = input.readEnumSet(MLClusterLevelStat.class);
        }
        if (input.readBoolean()) {
            nodeLevelStats = input.readEnumSet(MLNodeLevelStat.class);
        }
        if (input.readBoolean()) {
            algorithms = input.readEnumSet(FunctionName.class);
        }
        if (input.readBoolean()) {
            nodeIds = new HashSet<>(input.readStringList());
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        writeEnumSet(out, targetStatLevels);
        writeEnumSet(out, clusterLevelStats);
        writeEnumSet(out, nodeLevelStats);
        writeEnumSet(out, algorithms);
        out.writeOptionalStringCollection(nodeIds);
    }

    private void writeEnumSet(StreamOutput out, EnumSet set) throws IOException {
        if (set != null && set.size() > 0) {
            out.writeBoolean(true);
            out.writeEnumSet(set);
        } else {
            out.writeBoolean(false);
        }
    }

    // @Override
    // public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
    // builder.startObject();
    // builder.field(TARGET_STAT_LEVEL, targetStatLevels);
    // builder.field(CLUSTER_LEVEL_STATS, clusterLevelStats);
    // builder.field(NODE_LEVEL_STATS, nodeLevelStats);
    // builder.field(NODE_IDS, nodeIds);
    // builder.field(ALGORITHMS, algorithms);
    // builder.endObject();
    // return builder;
    // }

    public static MLStatsInput parse(XContentParser parser) throws IOException {
        EnumSet<MLStatLevel> targetStatLevels = EnumSet.noneOf(MLStatLevel.class);
        ;
        EnumSet<MLClusterLevelStat> clusterLevelStats = EnumSet.noneOf(MLClusterLevelStat.class);
        ;
        EnumSet<MLNodeLevelStat> nodeLevelStats = EnumSet.noneOf(MLNodeLevelStat.class);
        ;
        EnumSet<FunctionName> algorithms = EnumSet.noneOf(FunctionName.class);
        Set<String> nodeIds = new HashSet<>();

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case TARGET_STAT_LEVEL:
                    parseField(parser, targetStatLevels, input -> MLStatLevel.from(input.toUpperCase(Locale.ROOT)));
                    break;
                case CLUSTER_LEVEL_STATS:
                    parseField(parser, clusterLevelStats, input -> MLClusterLevelStat.from(input.toUpperCase(Locale.ROOT)));
                    break;
                case NODE_LEVEL_STATS:
                    parseField(parser, nodeLevelStats, input -> MLNodeLevelStat.from(input.toUpperCase(Locale.ROOT)));
                    break;
                case ALGORITHMS:
                    parseField(parser, algorithms, input -> FunctionName.from(input.toUpperCase(Locale.ROOT)));
                    break;
                case NODE_IDS:
                    parseField(parser, nodeIds);
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return MLStatsInput
            .builder()
            .targetStatLevels(targetStatLevels)
            .clusterLevelStats(clusterLevelStats)
            .nodeLevelStats(nodeLevelStats)
            .nodeIds(nodeIds)
            .algorithms(algorithms)
            .build();
    }

    private static void parseField(XContentParser parser, Set<?> set) throws IOException {
        parseField(parser, set, null);
    }

    private static <T> void parseField(XContentParser parser, Set<T> set, Function<String, T> function) throws IOException {
        ensureExpectedToken(XContentParser.Token.START_ARRAY, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
            String value = parser.text();
            if (function != null) {
                set.add(function.apply(value));
            } else {
                set.add((T) value);
            }
        }
    }

}
