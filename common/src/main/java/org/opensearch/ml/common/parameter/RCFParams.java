/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.parameter;

import lombok.Builder;
import lombok.Data;
import org.opensearch.common.ParseField;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.common.annotation.MLAlgoParameter;

import java.io.IOException;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;

@Data
@MLAlgoParameter(algorithms={FunctionName.RCF})
public class RCFParams implements MLAlgoParams{
    public static final String PARSE_FIELD_NAME = FunctionName.RCF.name();
    public static final NamedXContentRegistry.Entry XCONTENT_REGISTRY = new NamedXContentRegistry.Entry(
            MLAlgoParams.class,
            new ParseField(PARSE_FIELD_NAME),
            it -> parse(it)
    );

    public static final String SHINGLE_SIZE = "shingle_size";
    public static final String TIME_FIELD = "time_field";
    public static final String DATE_FORMAT = "date_format";
    public static final String TIME_ZONE = "time_zone";
    public static final String TRAINING_DATA_SIZE = "training_data_size";
    private Integer shingleSize;
    private String timeField;
    private String dateFormat;
    private String timeZone;
    private Integer trainingDataSize;

    @Builder
    public RCFParams(Integer shingleSize, String timeField, String dateFormat, String timeZone, Integer trainingDataSize) {
        this.shingleSize = shingleSize;
        this.timeField = timeField;
        this.dateFormat = dateFormat;
        this.timeZone = timeZone;
        this.trainingDataSize = trainingDataSize;
    }

    public RCFParams(StreamInput in) throws IOException {
        this.shingleSize = in.readOptionalInt();
        this.timeField = in.readOptionalString();
        this.dateFormat = in.readOptionalString();
        this.timeZone = in.readOptionalString();
        this.shingleSize = in.readOptionalInt();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeOptionalInt(shingleSize);
        out.writeOptionalString(timeField);
        out.writeOptionalString(dateFormat);
        out.writeOptionalString(timeZone);
        out.writeOptionalInt(shingleSize);
    }

    public static RCFParams parse(XContentParser parser) throws IOException {
        Integer shingleSize = null;
        String timeField = null;
        String dateFormat = null;
        String timeZone = null;
        Integer trainingDataSize = null;

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case SHINGLE_SIZE:
                    shingleSize = parser.intValue();
                    break;
                case TIME_FIELD:
                    timeField = parser.text();
                    break;
                case DATE_FORMAT:
                    dateFormat = parser.text();
                    break;
                case TIME_ZONE:
                    timeZone = parser.text();
                    break;
                case TRAINING_DATA_SIZE:
                    trainingDataSize = parser.intValue();
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return new RCFParams(shingleSize, timeField, dateFormat, timeZone, trainingDataSize);
    }

    @Override
    public int getVersion() {
        return 1;
    }

    @Override
    public String getWriteableName() {
        return PARSE_FIELD_NAME;
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        if (shingleSize != null) {
            builder.field(SHINGLE_SIZE, shingleSize);
        }
        if (timeField != null) {
            builder.field(TIME_FIELD, timeField);
        }
        if (dateFormat != null) {
            builder.field(DATE_FORMAT, dateFormat);
        }
        if (timeZone != null) {
            builder.field(TIME_ZONE, timeZone);
        }
        if (trainingDataSize != null) {
            builder.field(TRAINING_DATA_SIZE, trainingDataSize);
        }
        builder.endObject();
        return builder;
    }
}
