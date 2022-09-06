/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom.uploadchunk;

import org.junit.Before;
import org.junit.Test;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.io.stream.StreamOutput;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Arrays;

import static org.junit.Assert.*;

public class MLCustomModelChunkUploadRequestTest {

    private MLUploadChunkInput mlUploadChunkInput;

    @Before
    public void setUp() {
        byte[] content = new byte[] { (byte)0xe0, 0x4f, (byte)0xd0,
                0x20, (byte)0xea, 0x3a, 0x69, 0x10, (byte)0xa2, (byte)0xd8, 0x08, 0x00, 0x2b,
                0x30, 0x30, (byte)0x9d };
        mlUploadChunkInput = new MLUploadChunkInput("test_model", 1, content, 0, 1);
    }

    @Test
    public void writeTo_Success() throws IOException {
        MLUploadModelChunkRequest mlUploadModelChunkRequest = MLUploadModelChunkRequest.builder()
                .mlUploadInput(mlUploadChunkInput)
                .build();
        BytesStreamOutput bytesStreamOutput = new BytesStreamOutput();
        mlUploadModelChunkRequest.writeTo(bytesStreamOutput);
        MLUploadModelChunkRequest parsedModel = new MLUploadModelChunkRequest(bytesStreamOutput.bytes().streamInput());
        MLUploadChunkInput parsedInput = parsedModel.getMlUploadInput();
        assertEquals(parsedInput.getName(), mlUploadChunkInput.getName());
        assertEquals(parsedInput.getVersion(), mlUploadChunkInput.getVersion());
        assertEquals(parsedInput.getChunkNumber(), mlUploadChunkInput.getChunkNumber());
        assertEquals(parsedInput.getTotalChunks(), mlUploadChunkInput.getTotalChunks());
        assertTrue(Arrays.equals(parsedInput.getContent(), mlUploadChunkInput.getContent()));
    }

    @Test
    public void validate_Exception_NullModelId() {
        MLUploadModelChunkRequest mlUploadChunkInput = MLUploadModelChunkRequest.builder().build();

        ActionRequestValidationException exception = mlUploadChunkInput.validate();
        assertEquals("Validation Failed: 1: ML input can't be null;", exception.getMessage());
    }

    @Test
    public void fromActionRequest_Success() {
        MLUploadModelChunkRequest mlModelGetRequest = MLUploadModelChunkRequest.builder()
                .mlUploadInput(mlUploadChunkInput).build();
        ActionRequest actionRequest = new ActionRequest() {
            @Override
            public ActionRequestValidationException validate() {
                return null;
            }

            @Override
            public void writeTo(StreamOutput out) throws IOException {
                mlModelGetRequest.writeTo(out);
            }
        };
        MLUploadModelChunkRequest result = MLUploadModelChunkRequest.fromActionRequest(actionRequest);
        MLUploadChunkInput resultInput = result.getMlUploadInput();
        MLUploadChunkInput requestInput = mlModelGetRequest.getMlUploadInput();
        assertNotSame(result, mlModelGetRequest);
        assertEquals(resultInput.getName(), requestInput.getName());
        assertEquals(resultInput.getVersion(), requestInput.getVersion());
        assertEquals(resultInput.getChunkNumber(), requestInput.getChunkNumber());
        assertEquals(resultInput.getTotalChunks(), requestInput.getTotalChunks());
        assertTrue(Arrays.equals(resultInput.getContent(), requestInput.getContent()));
    }

    @Test(expected = UncheckedIOException.class)
    public void fromActionRequest_IOException() {
        ActionRequest actionRequest = new ActionRequest() {
            @Override
            public ActionRequestValidationException validate() {
                return null;
            }

            @Override
            public void writeTo(StreamOutput out) throws IOException {
                throw new IOException("test");
            }
        };
        MLUploadModelChunkRequest.fromActionRequest(actionRequest);
    }
}
