/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.stats;

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
import org.opensearch.common.xcontent.ToXContentObject;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;

@Getter
public class MLStatsInput implements ToXContentObject, Writeable {
    public static final String TARGET_STAT_LEVEL = "target_stat_levels";
    public static final String CLUSTER_LEVEL_STATS = "cluster_level_stats";
    public static final String NODE_LEVEL_STATS = "node_level_stats";
    public static final String ACTION_LEVEL_STATS = "action_level_stats";
    public static final String NODE_IDS = "node_ids";
    public static final String ALGORITHMS = "algorithms";
    public static final String ACTIONS = "actions";

    private EnumSet<MLStatLevel> targetStatLevels;
    private EnumSet<MLClusterLevelStat> clusterLevelStats;
    private EnumSet<MLNodeLevelStat> nodeLevelStats;
    private EnumSet<MLActionLevelStat> actionLevelStats;
    private EnumSet<FunctionName> algorithms;
    private EnumSet<ActionName> actions;
    private Set<String> nodeIds;

    @Builder
    public MLStatsInput(
        EnumSet<MLStatLevel> targetStatLevels,
        EnumSet<MLClusterLevelStat> clusterLevelStats,
        EnumSet<MLNodeLevelStat> nodeLevelStats,
        EnumSet<MLActionLevelStat> actionLevelStats,
        Set<String> nodeIds,
        EnumSet<FunctionName> algorithms,
        EnumSet<ActionName> actions
    ) {
        this.targetStatLevels = targetStatLevels;
        this.clusterLevelStats = clusterLevelStats;
        this.nodeLevelStats = nodeLevelStats;
        this.actionLevelStats = actionLevelStats;
        this.nodeIds = nodeIds;
        this.algorithms = algorithms;
        this.actions = actions;
    }

    public MLStatsInput() {
        this.targetStatLevels = EnumSet.noneOf(MLStatLevel.class);
        this.clusterLevelStats = EnumSet.noneOf(MLClusterLevelStat.class);
        this.nodeLevelStats = EnumSet.noneOf(MLNodeLevelStat.class);
        this.actionLevelStats = EnumSet.noneOf(MLActionLevelStat.class);
        this.nodeIds = new HashSet<>();
        this.algorithms = EnumSet.noneOf(FunctionName.class);
        this.actions = EnumSet.noneOf(ActionName.class);
    }

    public MLStatsInput(StreamInput input) throws IOException {
        targetStatLevels = input.readBoolean() ? input.readEnumSet(MLStatLevel.class) : EnumSet.noneOf(MLStatLevel.class);
        // if (input.readBoolean()) {
        // clusterLevelStats = input.readEnumSet(MLClusterLevelStat.class);
        // } else {
        // clusterLevelStats = EnumSet.noneOf(MLClusterLevelStat.class);
        // }
        clusterLevelStats = input.readBoolean() ? input.readEnumSet(MLClusterLevelStat.class) : EnumSet.noneOf(MLClusterLevelStat.class);
        // if (input.readBoolean()) {
        // nodeLevelStats = input.readEnumSet(MLNodeLevelStat.class);
        // } else {
        // nodeLevelStats = EnumSet.noneOf(MLNodeLevelStat.class);
        // }
        nodeLevelStats = input.readBoolean() ? input.readEnumSet(MLNodeLevelStat.class) : EnumSet.noneOf(MLNodeLevelStat.class);
        // if (input.readBoolean()) {
        // actionLevelStats = input.readEnumSet(MLActionLevelStat.class);
        // } else {
        // actionLevelStats = EnumSet.noneOf(MLActionLevelStat.class);
        // }
        actionLevelStats = input.readBoolean() ? input.readEnumSet(MLActionLevelStat.class) : EnumSet.noneOf(MLActionLevelStat.class);
        // if (input.readBoolean()) {
        // nodeIds = new HashSet<>(input.readStringList());
        // } else {
        // nodeIds = new HashSet<>();
        // }
        nodeIds = input.readBoolean() ? new HashSet<>(input.readStringList()) : new HashSet<>();
        // if (input.readBoolean()) {
        // algorithms = input.readEnumSet(FunctionName.class);
        // } else {
        // algorithms = EnumSet.noneOf(FunctionName.class);
        // }
        algorithms = input.readBoolean() ? input.readEnumSet(FunctionName.class) : EnumSet.noneOf(FunctionName.class);
        // if (input.readBoolean()) {
        // actions = input.readEnumSet(ActionName.class);
        // } else {
        // actions = EnumSet.noneOf(ActionName.class);
        // }
        actions = input.readBoolean() ? input.readEnumSet(ActionName.class) : EnumSet.noneOf(ActionName.class);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        writeEnumSet(out, targetStatLevels);
        writeEnumSet(out, clusterLevelStats);
        writeEnumSet(out, nodeLevelStats);
        writeEnumSet(out, actionLevelStats);
        out.writeOptionalStringCollection(nodeIds);
        writeEnumSet(out, algorithms);
        writeEnumSet(out, actions);
    }

    private void writeEnumSet(StreamOutput out, EnumSet set) throws IOException {
        if (set != null && set.size() > 0) {
            out.writeBoolean(true);
            out.writeEnumSet(set);
        } else {
            out.writeBoolean(false);
        }
    }

    public static MLStatsInput parse(XContentParser parser) throws IOException {
        EnumSet<MLStatLevel> targetStatLevels = EnumSet.noneOf(MLStatLevel.class);
        EnumSet<MLClusterLevelStat> clusterLevelStats = EnumSet.noneOf(MLClusterLevelStat.class);
        EnumSet<MLNodeLevelStat> nodeLevelStats = EnumSet.noneOf(MLNodeLevelStat.class);
        EnumSet<MLActionLevelStat> actionLevelStats = EnumSet.noneOf(MLActionLevelStat.class);
        Set<String> nodeIds = new HashSet<>();
        EnumSet<FunctionName> algorithms = EnumSet.noneOf(FunctionName.class);
        EnumSet<ActionName> actions = EnumSet.noneOf(ActionName.class);

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
                case ACTION_LEVEL_STATS:
                    parseField(parser, actionLevelStats, input -> MLActionLevelStat.from(input.toUpperCase(Locale.ROOT)));
                    break;
                case NODE_IDS:
                    parseField(parser, nodeIds);
                    break;
                case ALGORITHMS:
                    parseField(parser, algorithms, input -> FunctionName.from(input.toUpperCase(Locale.ROOT)));
                    break;
                case ACTIONS:
                    parseField(parser, actions, input -> ActionName.from(input.toUpperCase(Locale.ROOT)));
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
            .actionLevelStats(actionLevelStats)
            .nodeIds(nodeIds)
            .algorithms(algorithms)
            .actions(actions)
            .build();
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(TARGET_STAT_LEVEL, targetStatLevels);
        builder.field(CLUSTER_LEVEL_STATS, clusterLevelStats);
        builder.field(NODE_LEVEL_STATS, nodeLevelStats);
        builder.field(ACTION_LEVEL_STATS, actionLevelStats);
        builder.field(NODE_IDS, nodeIds);
        builder.field(ALGORITHMS, algorithms);
        builder.field(ACTIONS, actions);
        builder.endObject();
        return builder;
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
