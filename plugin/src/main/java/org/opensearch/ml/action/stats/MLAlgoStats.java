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

public class MLAlgoStats implements ToXContentFragment, Writeable {

    private Map<String, Object> algoActionStat; // { request_count: 1}

    public MLAlgoStats(StreamInput in) throws IOException {
        this.algoActionStat = in.readMap(StreamInput::readString, StreamInput::readGenericValue);
    }

    public MLAlgoStats(Map<String, Object> algoActionStatMap) {
        this.algoActionStat = algoActionStatMap;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeMap(algoActionStat, StreamOutput::writeString, StreamOutput::writeGenericValue);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        if (algoActionStat != null && algoActionStat.size() > 0) {
            for (Map.Entry<String, Object> entry : algoActionStat.entrySet()) {
                builder.field(entry.getKey(), entry.getValue());
            }
        }
        return builder;
    }
}
