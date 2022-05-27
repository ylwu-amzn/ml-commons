/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.stats;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;

import lombok.Getter;

import org.opensearch.action.support.nodes.BaseNodeResponse;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.ToXContentFragment;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.stats.MLAlgoActionStats;
import org.opensearch.ml.stats.MLNodeLevelStat;

public class MLStatsNodeResponse extends BaseNodeResponse implements ToXContentFragment {
    @Getter
    private Map<MLNodeLevelStat, Object> nodeStats;// node level stats
    @Getter
    private Map<FunctionName, MLAlgoActionStats> algorithmStats; // kmeans -> { train -> { request_count: 1} }

    /**
     * Constructor
     *
     * @param in StreamInput
     * @throws IOException throws an IO exception if the StreamInput cannot be reML from
     */
    public MLStatsNodeResponse(StreamInput in) throws IOException {
        super(in);
        this.nodeStats = in.readMap(stream -> stream.readEnum(MLNodeLevelStat.class), StreamInput::readGenericValue);
        this.algorithmStats = in.readMap(stream -> stream.readEnum(FunctionName.class), MLAlgoActionStats::new);
    }

    public MLStatsNodeResponse(DiscoveryNode node, Map<MLNodeLevelStat, Object> nodeStats) {
        super(node);
        this.nodeStats = nodeStats;
    }

    public MLStatsNodeResponse(
        DiscoveryNode node,
        Map<MLNodeLevelStat, Object> nodeStats,
        Map<FunctionName, MLAlgoActionStats> algorithmStats
    ) {
        super(node);
        this.nodeStats = nodeStats;
        this.algorithmStats = algorithmStats;
    }

    public boolean isEmpty() {
        return (nodeStats == null || nodeStats.size() == 0) && (algorithmStats == null || algorithmStats.size() == 0);
    }

    /**
     * Creates a new MLStatsNodeResponse object and reMLs in the stats from an input stream
     *
     * @param in StreamInput to reML from
     * @return MLStatsNodeResponse object corresponding to the input stream
     * @throws IOException throws an IO exception if the StreamInput cannot be reML from
     */
    public static MLStatsNodeResponse readStats(StreamInput in) throws IOException {
        return new MLStatsNodeResponse(in);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeMap(nodeStats, (stream, v) -> stream.writeEnum(v), StreamOutput::writeGenericValue);
        out.writeMap(algorithmStats, (stream, v) -> stream.writeEnum(v), (stream, stats) -> stats.writeTo(stream));
    }

    /**
     * Converts statsMap to xContent
     *
     * @param builder XContentBuilder
     * @param params Params
     * @return XContentBuilder
     * @throws IOException thrown by builder for invalid field
     */
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        for (Map.Entry<MLNodeLevelStat, Object> stat : nodeStats.entrySet()) {
            builder.field(stat.getKey().name().toLowerCase(Locale.ROOT), stat.getValue());
        }
        if (algorithmStats != null && algorithmStats.size() > 0) {
            builder.startObject("algorithms");
            for (Map.Entry<FunctionName, MLAlgoActionStats> stat : algorithmStats.entrySet()) {
                builder.startObject(stat.getKey().name().toLowerCase(Locale.ROOT));
                stat.getValue().toXContent(builder, params);
                builder.endObject();
            }
            builder.endObject();
        }
        return builder;
    }
}
