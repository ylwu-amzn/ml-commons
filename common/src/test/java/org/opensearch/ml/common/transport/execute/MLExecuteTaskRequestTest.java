/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.execute;

import org.junit.Test;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.ml.common.parameter.FunctionName;
import org.opensearch.ml.common.parameter.Input;
import org.opensearch.ml.common.parameter.LocalSampleCalculatorInput;
import org.opensearch.ml.common.transport.model.MLModelDeleteRequest;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;

public class MLExecuteTaskRequestTest {

    @Test
    public void writeTo_Success() throws IOException {
        String operation = "sum";
        LocalSampleCalculatorInput input = LocalSampleCalculatorInput.builder().operation(operation).inputData(Arrays.asList(1.0, 2.0, 3.0)).build();
        MLExecuteTaskRequest request = MLExecuteTaskRequest.builder().functionName(FunctionName.LOCAL_SAMPLE_CALCULATOR).input(input).build();
        BytesStreamOutput bytesStreamOutput = new BytesStreamOutput();
        request.writeTo(bytesStreamOutput);

        MLExecuteTaskRequest parsedRequest = new MLExecuteTaskRequest(bytesStreamOutput.bytes().streamInput());
        LocalSampleCalculatorInput parsedInput = (LocalSampleCalculatorInput)parsedRequest.getInput();
        assertEquals(operation, parsedInput.getClass());
    }
}
