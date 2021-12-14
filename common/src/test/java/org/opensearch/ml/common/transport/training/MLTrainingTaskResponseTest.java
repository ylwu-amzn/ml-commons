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

package org.opensearch.ml.common.transport.training;

import java.io.IOException;
import java.io.UncheckedIOException;

import org.junit.Test;
import org.opensearch.action.ActionResponse;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.ml.common.output.MLTrainingOutput;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;

public class MLTrainingTaskResponseTest {

    @Test
    public void writeTo() throws IOException {
        MLTrainingOutput output = MLTrainingOutput.builder().status("success")
                .modelId("taskId").build();
        MLTrainingTaskResponse response = MLTrainingTaskResponse.builder()
                .output(output)
                .build();
        BytesStreamOutput bytesStreamOutput = new BytesStreamOutput();
        response.writeTo(bytesStreamOutput);
        response = new MLTrainingTaskResponse(bytesStreamOutput.bytes().streamInput());
        MLTrainingOutput modelTrainingOutput = (MLTrainingOutput)response.getOutput();
        assertEquals("success", modelTrainingOutput.getStatus());
        assertEquals("taskId", modelTrainingOutput.getModelId());
    }

    @Test
    public void fromActionResponse_Success_WithMLTrainingTaskResponse() {
        MLTrainingOutput output = MLTrainingOutput.builder().status("success")
                .modelId("taskId").build();
        MLTrainingTaskResponse response = MLTrainingTaskResponse.builder()
                .output(output)
                .build();
        assertSame(response, MLTrainingTaskResponse.fromActionResponse(response));
    }

    @Test
    public void fromActionResponse_Success_WithNonMLTrainingTaskResponse() {
        MLTrainingOutput output = MLTrainingOutput.builder().status("success")
                .modelId("taskId").build();
        MLTrainingTaskResponse response = MLTrainingTaskResponse.builder()
                .output(output)
                .build();
        ActionResponse actionResponse = new ActionResponse() {
            @Override
            public void writeTo(StreamOutput out) throws IOException {
                response.writeTo(out);
            }
        };

        MLTrainingTaskResponse result = MLTrainingTaskResponse.fromActionResponse(actionResponse);
        assertNotSame(response, result);
        MLTrainingOutput modelTrainingOutput = (MLTrainingOutput)response.getOutput();
        MLTrainingOutput resultModelTrainingOutput = (MLTrainingOutput)result.getOutput();
        assertEquals(modelTrainingOutput.getStatus(), resultModelTrainingOutput.getStatus());
        assertEquals(modelTrainingOutput.getModelId(), resultModelTrainingOutput.getModelId());
    }

    @Test(expected = UncheckedIOException.class)
    public void fromActionResponse_Exception() {
        ActionResponse actionResponse = new ActionResponse() {
            @Override
            public void writeTo(StreamOutput out) throws IOException {
                throw new IOException("test");
            }
        };

        MLTrainingTaskResponse.fromActionResponse(actionResponse);
    }
}