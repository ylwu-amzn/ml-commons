/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.model;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;

import java.io.IOException;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;

@Setter
@Getter
public class MetricsCorrelationModelConfig extends MLModelConfig {

    public static final String PARSE_FIELD_NAME = FunctionName.METRICS_CORRELATION.name();

    @Builder(toBuilder = true)
    public MetricsCorrelationModelConfig(String modelType, String allConfig) {
        super(modelType, allConfig);
    }

    @Override
    public String getWriteableName() {
        return PARSE_FIELD_NAME;
    }

    public MetricsCorrelationModelConfig(StreamInput in) throws IOException{
        super(in);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        if (modelType != null) {
            builder.field(MODEL_TYPE_FIELD, modelType);
        }
        if (allConfig != null) {
            builder.field(ALL_CONFIG_FIELD, allConfig);
        }
        builder.endObject();
        return builder;
    }

    public static MetricsCorrelationModelConfig parse(XContentParser parser) throws IOException {
        String modelType = null;
        String allConfig = null;

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case MODEL_TYPE_FIELD:
                    modelType = parser.text();
                    break;
                case ALL_CONFIG_FIELD:
                    allConfig = parser.text();
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return new MetricsCorrelationModelConfig(modelType, allConfig);
    }
}
