/*
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  The OpenSearch Contributors require contributions made to
 *  this file be licensed under the Apache-2.0 license or a
 *  compatible open source license.
 *
 *  Modifications Copyright OpenSearch Contributors. See
 *  GitHub history for details.
 */

package org.opensearch.ml.common.output;

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

@Data
@EqualsAndHashCode(callSuper=false)
@MLAlgoOutput(MLOutputType.SAMPLE_ALGO)
public class SampleAlgoOutput extends MLOutput {

    private static final MLOutputType OUTPUT_TYPE = MLOutputType.SAMPLE_ALGO;
    public static final String SAMPLE_RESULT_FIELD = "sample_result";
    private Double sampleResult;

    @Builder
    public SampleAlgoOutput(Double sampleResult) {
        super(OUTPUT_TYPE);
        this.sampleResult = sampleResult;
    }

    public SampleAlgoOutput(StreamInput in) throws IOException {
        super(OUTPUT_TYPE);
        sampleResult = in.readOptionalDouble();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeOptionalDouble(sampleResult);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        if (sampleResult != null) {
            builder.field(SAMPLE_RESULT_FIELD, sampleResult);
        }

        builder.endObject();
        return builder;
    }

    @Override
    public MLOutputType getType() {
        return OUTPUT_TYPE;
    }
}
