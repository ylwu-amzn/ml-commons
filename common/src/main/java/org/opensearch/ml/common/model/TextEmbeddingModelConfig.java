/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.model;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.opensearch.common.ParseField;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;

import java.io.IOException;
import java.util.Locale;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;

@Setter
@Getter
public class TextEmbeddingModelConfig extends MLModelConfig {
    public static final String PARSE_FIELD_NAME = FunctionName.TEXT_EMBEDDING.name();
    public static final NamedXContentRegistry.Entry XCONTENT_REGISTRY = new NamedXContentRegistry.Entry(
            TextEmbeddingModelConfig.class,
            new ParseField(PARSE_FIELD_NAME),
            it -> parse(it)
    );

    public static final String EMBEDDING_DIMENSION_FIELD = "embedding_dimension";
    public static final String FRAMEWORK_TYPE_FIELD = "framework_type";
    public static final String POOLING_METHOD_FIELD = "pooling_method";
    public static final String NORMALIZE_RESULT_FIELD = "normalize_result";
    public static final String MODEL_MAX_LENGTH_FIELD = "model_max_length";
    public static final String DEFAULT_POOLING_METHOD = "mean";

    private final Integer embeddingDimension;
    private final FrameworkType frameworkType;
    private final String poolingMethod;
    private final boolean normalizeResult;
    private final Integer modelMaxLength;

    @Builder(toBuilder = true)
    public TextEmbeddingModelConfig(String modelType, Integer embeddingDimension, FrameworkType frameworkType, String allConfig,
                                    String poolingMethod, boolean normalizeResult, Integer modelMaxLength) {
        super(modelType, allConfig);
        if (embeddingDimension == null) {
            throw new IllegalArgumentException("embedding dimension is null");
        }
        if (frameworkType == null) {
            throw new IllegalArgumentException("framework type is null");
        }
        this.embeddingDimension = embeddingDimension;
        this.frameworkType = frameworkType;
        if (poolingMethod != null) {
            this.poolingMethod = poolingMethod;
        } else {
            this.poolingMethod = DEFAULT_POOLING_METHOD;
        }
        this.normalizeResult = normalizeResult;
        this.modelMaxLength = modelMaxLength;
    }

    public static TextEmbeddingModelConfig parse(XContentParser parser) throws IOException {
        String modelType = null;
        Integer embeddingDimension = null;
        FrameworkType frameworkType = null;
        String allConfig = null;
        String poolingMethod = DEFAULT_POOLING_METHOD;
        boolean normalizeResult = false;
        Integer modelMaxLength = null;

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case MODEL_TYPE_FIELD:
                    modelType = parser.text();
                    break;
                case EMBEDDING_DIMENSION_FIELD:
                    embeddingDimension = parser.intValue();
                    break;
                case FRAMEWORK_TYPE_FIELD:
                    frameworkType = FrameworkType.from(parser.text().toUpperCase(Locale.ROOT));
                    break;
                case ALL_CONFIG_FIELD:
                    allConfig = parser.text();
                    break;
                case POOLING_METHOD_FIELD:
                    poolingMethod = parser.text();
                    break;
                case NORMALIZE_RESULT_FIELD:
                    normalizeResult = parser.booleanValue();
                    break;
                case MODEL_MAX_LENGTH_FIELD:
                    modelMaxLength = parser.intValue();
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return new TextEmbeddingModelConfig(modelType,  embeddingDimension, frameworkType, allConfig, poolingMethod, normalizeResult, modelMaxLength);
    }

    @Override
    public String getWriteableName() {
        return PARSE_FIELD_NAME;
    }

    public TextEmbeddingModelConfig(StreamInput in) throws IOException{
        super(in);
        embeddingDimension = in.readInt();
        frameworkType = in.readEnum(FrameworkType.class);
        poolingMethod = in.readString();
        normalizeResult = in.readBoolean();
        modelMaxLength = in.readOptionalInt();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeInt(embeddingDimension);
        out.writeEnum(frameworkType);
        out.writeString(poolingMethod);
        out.writeBoolean(normalizeResult);
        out.writeOptionalInt(modelMaxLength);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        if (modelType != null) {
            builder.field(MODEL_TYPE_FIELD, modelType);
        }
        if (embeddingDimension != null) {
            builder.field(EMBEDDING_DIMENSION_FIELD, embeddingDimension);
        }
        if (frameworkType != null) {
            builder.field(FRAMEWORK_TYPE_FIELD, frameworkType);
        }
        if (allConfig != null) {
            builder.field(ALL_CONFIG_FIELD, allConfig);
        }
        if (modelMaxLength != null) {
            builder.field(MODEL_MAX_LENGTH_FIELD, modelMaxLength);
        }
        builder.field(POOLING_METHOD_FIELD, poolingMethod);
        builder.field(NORMALIZE_RESULT_FIELD, normalizeResult);
        builder.endObject();
        return builder;
    }

    public enum FrameworkType {
        HUGGINGFACE_TRANSFORMERS,
        SENTENCE_TRANSFORMERS,
        HUGGINGFACE_TRANSFORMERS_NEURON;

        public static FrameworkType from(String value) {
            try {
                return FrameworkType.valueOf(value);
            } catch (Exception e) {
                throw new IllegalArgumentException("Wrong framework type");
            }
        }
    }

}
