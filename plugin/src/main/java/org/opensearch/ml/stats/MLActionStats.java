/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.stats;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;

import lombok.Getter;

import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.common.xcontent.ToXContentFragment;
import org.opensearch.common.xcontent.XContentBuilder;

public class MLActionStats implements ToXContentFragment, Writeable {

    @Getter
    private Map<MLActionLevelStat, Object> algoActionStat; // { request_count: 1}

    public MLActionStats(StreamInput in) throws IOException {
        this.algoActionStat = in.readMap(stream -> stream.readEnum(MLActionLevelStat.class), StreamInput::readGenericValue);
    }

    public MLActionStats(Map<MLActionLevelStat, Object> algoActionStatMap) {
        this.algoActionStat = algoActionStatMap;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeMap(algoActionStat, (stream, v) -> stream.writeEnum(v), StreamOutput::writeGenericValue);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        if (algoActionStat != null && algoActionStat.size() > 0) {
            for (Map.Entry<MLActionLevelStat, Object> entry : algoActionStat.entrySet()) {
                builder.field(entry.getKey().name().toLowerCase(Locale.ROOT), entry.getValue());
            }
        }
        return builder;
    }
}
