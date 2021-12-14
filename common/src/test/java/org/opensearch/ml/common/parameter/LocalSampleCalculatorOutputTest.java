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

package org.opensearch.ml.common.parameter;

import org.junit.Before;
import org.junit.Test;
import org.opensearch.common.Strings;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.ml.common.output.LocalSampleCalculatorOutput;

import java.io.IOException;

import static org.junit.Assert.assertEquals;

public class LocalSampleCalculatorOutputTest {

    LocalSampleCalculatorOutput output;
    @Before
    public void setUp() {
        output = LocalSampleCalculatorOutput.builder()
                .totalSum(1.0)
                .build();
    }

    @Test
    public void toXContent() throws IOException {
        XContentBuilder builder = XContentFactory.contentBuilder(XContentType.JSON);
        output.toXContent(builder, ToXContent.EMPTY_PARAMS);
        String jsonStr = Strings.toString(builder);
        assertEquals("{\"result\":1.0}", jsonStr);
    }

    @Test
    public void toXContent_EmptyOutput() throws IOException {
        LocalSampleCalculatorOutput output = LocalSampleCalculatorOutput.builder().build();
        XContentBuilder builder = XContentFactory.contentBuilder(XContentType.JSON);
        output.toXContent(builder, ToXContent.EMPTY_PARAMS);
        String jsonStr = Strings.toString(builder);
        assertEquals("{}", jsonStr);
    }

    @Test
    public void readInputStream_Success() throws IOException {
        readInputStream(output);
    }

    private void readInputStream(LocalSampleCalculatorOutput output) throws IOException {
        BytesStreamOutput bytesStreamOutput = new BytesStreamOutput();
        output.writeTo(bytesStreamOutput);

        StreamInput streamInput = bytesStreamOutput.bytes().streamInput();
        LocalSampleCalculatorOutput parsedOutput = new LocalSampleCalculatorOutput(streamInput);
        assertEquals(output, parsedOutput);
    }
}
