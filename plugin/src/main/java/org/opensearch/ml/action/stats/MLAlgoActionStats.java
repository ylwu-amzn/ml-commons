/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.stats;

import java.io.IOException;
import java.util.Map;

import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.common.xcontent.ToXContentFragment;
import org.opensearch.common.xcontent.XContentBuilder;

public class MLAlgoActionStats implements ToXContentFragment, Writeable {

    private Map<String, MLAlgoStats> algoStats; // train -> { request_count: 1}

    public MLAlgoActionStats(StreamInput in) throws IOException {
        this.algoStats = in.readMap(StreamInput::readString, MLAlgoStats::new);
    }

    public MLAlgoActionStats(Map<String, MLAlgoStats> algoStats) {
        this.algoStats = algoStats;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeMap(algoStats, StreamOutput::writeString, (stream, stats) -> stats.writeTo(stream));
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        if (algoStats != null && algoStats.size() > 0) {
            for (Map.Entry<String, MLAlgoStats> entry : algoStats.entrySet()) {
                builder.startObject(entry.getKey());
                entry.getValue().toXContent(builder, params);
                builder.endObject();
            }
        }
        return builder;
    }
}
