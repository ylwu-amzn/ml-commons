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
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.common.MLCommonsClassLoader;
import org.opensearch.ml.common.annotation.FunctionInput;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataframe.DefaultDataFrame;
import org.opensearch.ml.common.input.dataset.DataFrameInputDataset;
import org.opensearch.ml.common.input.dataset.MLInputDataType;
import org.opensearch.ml.common.input.dataset.MLInputDataset;
import org.opensearch.ml.common.input.dataset.SearchQueryInputDataset;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.parameter.Parameters;
import org.opensearch.search.builder.SearchSourceBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;

/**
 * ML input data: algirithm name, parameters and input data set.
 */
@FunctionInput(functions={FunctionName.KMEANS,
        FunctionName.LINEAR_REGRESSION,
        FunctionName.SAMPLE_ALGO})
public class MLInput implements Input {

    public static final String ALGORITHM_FIELD = "algorithm";
    public static final String ML_PARAMETERS_FIELD = "parameters";
    public static final String INPUT_INDEX_FIELD = "input_index";
    public static final String INPUT_QUERY_FIELD = "input_query";
    public static final String INPUT_DATA_FIELD = "input_data";

    // Algorithm name
    private FunctionName algorithm;
    // ML algorithm parameters
    private Parameters parameters;
    // Input data to train model, run trained model to predict or run ML algorithms(no-model-based) directly.
    private MLInputDataset inputDataset;

    private int version = 1;

    @Builder(toBuilder = true)
    public MLInput(FunctionName algorithm, Parameters parameters, MLInputDataset inputDataset) {
        validate(algorithm, inputDataset);
        this.algorithm = algorithm;
        this.parameters = parameters;
        this.inputDataset = inputDataset;
    }

    private void validate(FunctionName algorithm, MLInputDataset inputDataset) {
        if (algorithm == null) {
            throw new IllegalArgumentException("algorithm can't be null");
        }
        if (inputDataset == null) {
            throw new IllegalArgumentException("inputDataset can't be null");
        }
    }

    public MLInput(FunctionName functionName, StreamInput in) throws IOException {
        //this.algorithm = in.readEnum(FunctionName.class);
        setFunctionName(functionName);
        if (in.readBoolean()) {
            this.parameters = MLCommonsClassLoader.initInstance(algorithm, in, StreamInput.class);
        }
        if (in.readBoolean()) {
            MLInputDataType inputDataType = in.readEnum(MLInputDataType.class);
            this.inputDataset = MLCommonsClassLoader.initInstance(inputDataType, in, StreamInput.class);
        }
        this.version = in.readInt();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        Input.super.writeTo(out);
        //out.writeEnum(algorithm);
        if (parameters != null) {
            out.writeBoolean(true);
            parameters.writeTo(out);
        } else {
            out.writeBoolean(false);
        }
        if (inputDataset != null) {
            out.writeBoolean(true);
            inputDataset.writeTo(out);
        } else {
            out.writeBoolean(false);
        }
        out.writeInt(version);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(ALGORITHM_FIELD, algorithm.name());
        if (parameters != null) {
            builder.field(ML_PARAMETERS_FIELD, parameters);
        }
        if (inputDataset != null) {
            switch (inputDataset.getInputDataType()) {
                case SEARCH_QUERY:
                    builder.field(INPUT_INDEX_FIELD, ((SearchQueryInputDataset)inputDataset).getIndices().toArray(new String[0]));
                    builder.field(INPUT_QUERY_FIELD, ((SearchQueryInputDataset)inputDataset).getSearchSourceBuilder());
                    break;
                case DATA_FRAME:
                    builder.startObject(INPUT_DATA_FIELD);
                    ((DataFrameInputDataset)inputDataset).getDataFrame().toXContent(builder, EMPTY_PARAMS);
                    builder.endObject();
                    break;
                default:
                    break;
            }

        }
        builder.endObject();
        return builder;
    }

    public static MLInput parse(XContentParser parser, String inputAlgoName) throws IOException {
        String algorithmName = inputAlgoName.toUpperCase(Locale.ROOT);
        FunctionName algorithm = FunctionName.valueOf(algorithmName);
        Parameters mlParameters = null;
        SearchSourceBuilder searchSourceBuilder = null;
        List<String> sourceIndices = new ArrayList<>();
        DataFrame dataFrame = null;

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case ML_PARAMETERS_FIELD:
                    mlParameters = parser.namedObject(Parameters.class, algorithmName, null);
                    break;
                case INPUT_INDEX_FIELD:
                    ensureExpectedToken(XContentParser.Token.START_ARRAY, parser.currentToken(), parser);
                    while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
                        sourceIndices.add(parser.text());
                    }
                    break;
                case INPUT_QUERY_FIELD:
                    ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
                    searchSourceBuilder = SearchSourceBuilder.fromXContent(parser, false);
                    break;
                case INPUT_DATA_FIELD:
                    dataFrame = DefaultDataFrame.parse(parser);
                default:
                    parser.skipChildren();
                    break;
            }
        }
        MLInputDataset inputDataset = createInputDataSet(searchSourceBuilder, sourceIndices, dataFrame);
        return new MLInput(algorithm, mlParameters, inputDataset);
    }

    private static MLInputDataset createInputDataSet(SearchSourceBuilder searchSourceBuilder, List<String> sourceIndices, DataFrame dataFrame) {
        if (dataFrame != null) {
            return new DataFrameInputDataset(dataFrame);
        }
        if (sourceIndices != null && searchSourceBuilder != null) {
            return new SearchQueryInputDataset(sourceIndices, searchSourceBuilder);
        }
        return null;
    }

    @Override
    public FunctionName getFunctionName() {
        return this.algorithm;
    }

    @Override
    public void setFunctionName(FunctionName functionName) {
        this.algorithm = functionName;
    }

    public DataFrame getDataFrame() {
        if (inputDataset == null || !(inputDataset instanceof DataFrameInputDataset)) {
            return null;
        }
        return ((DataFrameInputDataset)inputDataset).getDataFrame();
    }

}
