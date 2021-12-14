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

package org.opensearch.ml.common.input;

import lombok.Builder;
import lombok.Data;
import org.opensearch.common.ParseField;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.input.dataset.MLInputDataset;
import org.opensearch.ml.common.input.parameter.MLAlgoParams;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;

public class LocalSampleCalculatorInput implements Input {
    public static final String PARSE_FIELD_NAME = FunctionName.LOCAL_SAMPLE_CALCULATOR.name();
    public static final NamedXContentRegistry.Entry XCONTENT_REGISTRY = new NamedXContentRegistry.Entry(
            Input.class,
            new ParseField(PARSE_FIELD_NAME),
            it -> parse(it)
    );

    public static final String OPERATION_FIELD = "operation";
    public static final String INPUT_DATA_FIELD = "input_data";
    private FunctionName functionName;

    public static LocalSampleCalculatorInput parse(XContentParser parser) throws IOException {
        String operation = null;
        List<Double> inputData = new ArrayList<>();

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case OPERATION_FIELD:
                    operation = parser.text();
                    break;
                case INPUT_DATA_FIELD:
                    ensureExpectedToken(XContentParser.Token.START_ARRAY, parser.currentToken(), parser);
                    while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
                        inputData.add(parser.doubleValue());
                    }
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return new LocalSampleCalculatorInput(operation, inputData);
    }

    String operation;
    List<Double> inputData;

    @Builder
    public LocalSampleCalculatorInput(String operation, List<Double> inputData) {
        if (operation == null) {
            throw new IllegalArgumentException("wrong operation");
        }
        if (inputData == null || inputData.size() == 0) {
            throw new IllegalArgumentException("empty input data");
        }
        this.operation = operation;
        this.inputData = inputData;
        this.functionName = FunctionName.LOCAL_SAMPLE_CALCULATOR;
    }

    @Override
    public FunctionName getFunctionName() {
        return this.functionName;
    }

    @Override
    public void setFunctionName(FunctionName functionName) {
        this.functionName = functionName;
    }

    public LocalSampleCalculatorInput(StreamInput in) throws IOException {
        this.operation = in.readString();
        int size = in.readInt();
        this.inputData = new ArrayList<>();
        for (int i = 0; i<size; i++) {
            inputData.add(in.readDouble());
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(operation);
        out.writeInt(inputData.size());
        for (Double d : inputData) {
            out.writeDouble(d.doubleValue());
        }
    }

    @Override
    public MLInputDataset getInputDataset() {
        return Input.super.getInputDataset();
    }

    @Override
    public MLAlgoParams getParameters() {
        return Input.super.getParameters();
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(OPERATION_FIELD, operation);
        builder.field(INPUT_DATA_FIELD, inputData);
        builder.endObject();
        return builder;
    }
}
