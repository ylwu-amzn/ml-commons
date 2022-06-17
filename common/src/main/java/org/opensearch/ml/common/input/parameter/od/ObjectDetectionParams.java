/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.input.parameter.od;

import lombok.Builder;
import lombok.Data;
import org.opensearch.common.ParseField;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.annotation.MLAlgoParameter;
import org.opensearch.ml.common.input.parameter.MLAlgoParams;

import java.io.IOException;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;

@Data
@MLAlgoParameter(algorithms={FunctionName.OBJECT_DETECTION})
public class ObjectDetectionParams implements MLAlgoParams {
    public static final String PARSE_FIELD_NAME = FunctionName.OBJECT_DETECTION.name();
    public static final NamedXContentRegistry.Entry XCONTENT_REGISTRY = new NamedXContentRegistry.Entry(
            MLAlgoParams.class,
            new ParseField(PARSE_FIELD_NAME),
            it -> parse(it)
    );

    public static final String URL_FIELD = "url";
    private String url;

    @Builder
    public ObjectDetectionParams(String url) {
        this.url = url;
    }

    public ObjectDetectionParams(StreamInput in) throws IOException {
        this.url = in.readString();
    }

    public static MLAlgoParams parse(XContentParser parser) throws IOException {
        String url = null;

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case URL_FIELD:
                    url = parser.text();
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return new ObjectDetectionParams(url);
    }

    @Override
    public String getWriteableName() {
        return PARSE_FIELD_NAME;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(url);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        if (url != null) {
            builder.field(URL_FIELD, url);
        }
        builder.endObject();
        return builder;
    }

    @Override
    public int getVersion() {
        return 1;
    }
}
