/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 *
 */

package org.opensearch.ml.common.input;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.common.TestHelper;
import org.opensearch.ml.common.input.LocalSampleCalculatorInput;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class LocalSampleCalculatorInputTest {

    LocalSampleCalculatorInput input;

    Function<XContentParser, LocalSampleCalculatorInput> function = parser -> {
        try {
            return LocalSampleCalculatorInput.parse(parser);
        } catch (IOException e) {
            throw new RuntimeException("failed to parse LocalSampleCalculatorInput", e);
        }
    };

    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();

    @Before
    public void setUp() {
        List<Double> inputData = new ArrayList<>();
        inputData.add(1.0);
        inputData.add(2.0);
        inputData.add(3.0);
        input = LocalSampleCalculatorInput.builder()
                .operation("sum")
                .inputData(inputData)
                .build();
    }

    @Test
    public void constructor_NullOperation() {
        exceptionRule.expect(IllegalArgumentException.class);
        exceptionRule.expectMessage("wrong operation");
        LocalSampleCalculatorInput.builder().build();
    }

    @Test
    public void readInputStream_Success() throws IOException {
        BytesStreamOutput bytesStreamOutput = new BytesStreamOutput();
        input.writeTo(bytesStreamOutput);

        StreamInput streamInput = bytesStreamOutput.bytes().streamInput();
        LocalSampleCalculatorInput parsedParams = new LocalSampleCalculatorInput(streamInput);
        assertEquals(input, parsedParams);
    }

    @Test
    public void parse_LocalSampleCalculatorInput() throws IOException {
        TestHelper.testParse(input, function);
    }
}
