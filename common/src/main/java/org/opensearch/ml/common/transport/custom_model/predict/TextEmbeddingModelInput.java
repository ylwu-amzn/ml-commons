/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom_model.predict;

import lombok.Builder;
import lombok.Data;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;

/**
 * ML input data: algirithm name, parameters and input data set.
 */
@Data
public class TextEmbeddingModelInput implements CustomModelInput {

    public static final String MODEL_ID_FIELD = "model_id";
    public static final String DOCS_FIELD = "docs";
    public static final String TARGET_RESPONSE_FIELD = "target_response";

    private String modelId;
    private List<String> docs;
    private List<String> targetResponse;

    @Builder(toBuilder = true)
    public TextEmbeddingModelInput(String modelId, List<String> docs, List<String> targetResponse) {
        Objects.requireNonNull(modelId);
        Objects.requireNonNull(docs);
        if (docs.size() == 0) {
            throw new IllegalArgumentException("empty docs");
        }
        this.modelId = modelId;
        this.docs = docs;
        this.targetResponse = targetResponse;
    }


    public TextEmbeddingModelInput(StreamInput in) throws IOException {
        modelId = in.readString();
        docs = in.readStringList();
        targetResponse = in.readOptionalStringList();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(modelId);
        out.writeStringCollection(docs);
        out.writeOptionalStringCollection(targetResponse);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(MODEL_ID_FIELD, modelId);
        builder.field(DOCS_FIELD, docs.toArray());
        if (targetResponse != null && targetResponse.size() > 0) {
            builder.field(TARGET_RESPONSE_FIELD, targetResponse.toArray());
        }
        builder.endObject();
        return builder;
    }

    public static TextEmbeddingModelInput parse(XContentParser parser, String inputModelId) throws IOException {
        String modelId = inputModelId;
        List<String> docs = new ArrayList<>();
        List<String> targetResponse = new ArrayList<>();

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case MODEL_ID_FIELD:
                    modelId = parser.text();
                    break;
                case DOCS_FIELD:
                    ensureExpectedToken(XContentParser.Token.START_ARRAY, parser.currentToken(), parser);
                    while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
                        docs.add(parser.text());
                    }
                    break;
                case TARGET_RESPONSE_FIELD:
                    ensureExpectedToken(XContentParser.Token.START_ARRAY, parser.currentToken(), parser);
                    while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
                        targetResponse.add(parser.text());
                    }
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return new TextEmbeddingModelInput(modelId, docs, targetResponse);
    }


    public FunctionName getFunctionName() {
        return FunctionName.CUSTOM;
    }

}
