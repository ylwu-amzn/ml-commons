/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.output.od;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.ml.common.annotation.MLAlgoOutput;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.common.output.MLOutputType;

import java.io.IOException;
import java.util.List;

@Data
@EqualsAndHashCode(callSuper=false)
@MLAlgoOutput(MLOutputType.OBJECT_DETECTION)
public class ObjectDetectionOutput extends MLOutput {

    private static final MLOutputType OUTPUT_TYPE = MLOutputType.OBJECT_DETECTION;
    public static final String OBJECTS_FIELD = "sample_result";
    private String[] objects;

    @Builder
    public ObjectDetectionOutput(final List<String> objects) {
        super(OUTPUT_TYPE);
        this.objects = objects == null? new String[0] : objects.toArray(new String[0]);
    }

    public ObjectDetectionOutput(StreamInput in) throws IOException {
        super(OUTPUT_TYPE);
        objects = in.readOptionalStringArray();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeOptionalStringArray(objects);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        if (objects != null) {
            builder.field(OBJECTS_FIELD, objects);
        }
        builder.endObject();
        return builder;
    }

    @Override
    public MLOutputType getType() {
        return OUTPUT_TYPE;
    }
}
