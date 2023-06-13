/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.input.remote;

import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.common.utils.StringUtils.getParameterMap;

@org.opensearch.ml.common.annotation.MLInput(functionNames = {FunctionName.REMOTE})
public class RemoteInferenceMLInput extends MLInput {
    public static final String PARAMETERS_FIELD = "parameters";

    public RemoteInferenceMLInput(StreamInput in) throws IOException {
        super(in);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
    }

    public RemoteInferenceMLInput(XContentParser parser, FunctionName functionName) throws IOException {
        super();
        this.algorithm = functionName;
        Map<String, ?> parameterObjs = new HashMap<>();

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case PARAMETERS_FIELD:
                    parameterObjs = parser.map();
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        Map<String, String> parameters = getParameterMap(parameterObjs);
        inputDataset = new RemoteInferenceInputDataSet(parameters);
    }

}
