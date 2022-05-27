/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.stats;

import java.io.IOException;
import java.util.Map;

import lombok.Getter;

import org.opensearch.action.support.nodes.BaseNodeResponse;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.ToXContentFragment;
import org.opensearch.common.xcontent.XContentBuilder;

public class MLStatsNodeResponse extends BaseNodeResponse implements ToXContentFragment {
    @Getter
    private Map<String, Object> nodeStats;// node level stats
    @Getter
    private Map<String, MLAlgoActionStats> algorithmStats; // kmeans -> { train -> { request_count: 1} }

    /**
     * Constructor
     *
     * @param in StreamInput
     * @throws IOException throws an IO exception if the StreamInput cannot be reML from
     */
    public MLStatsNodeResponse(StreamInput in) throws IOException {
        super(in);
        this.nodeStats = in.readMap(StreamInput::readString, StreamInput::readGenericValue);
        this.nodeStats = in.readMap(StreamInput::readString, MLAlgoActionStats::new);
    }

    public MLStatsNodeResponse(DiscoveryNode node, Map<String, Object> nodeStats) {
        super(node);
        this.nodeStats = nodeStats;
    }

    public MLStatsNodeResponse(DiscoveryNode node, Map<String, Object> nodeStats, Map<String, MLAlgoActionStats> algorithmStats) {
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
        out.writeMap(nodeStats, StreamOutput::writeString, StreamOutput::writeGenericValue);
        out.writeMap(algorithmStats, StreamOutput::writeString, (stream, stats) -> stats.writeTo(stream));
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
        for (Map.Entry<String, Object> stat : nodeStats.entrySet()) {
            builder.field(stat.getKey(), stat.getValue());
        }
        if (algorithmStats != null && algorithmStats.size() > 0) {
            builder.startObject("algorithms");
            for (Map.Entry<String, MLAlgoActionStats> stat : algorithmStats.entrySet()) {
                builder.startObject(stat.getKey());
                stat.getValue().toXContent(builder, params);
                builder.endObject();
            }
            builder.endObject();
        }
        return builder;
    }
}
