/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom.uploadchunk;

import static org.opensearch.ml.common.TestHelper.parser;

import org.junit.Rule;
import org.junit.Test;
import org.junit.Assert;
import org.junit.rules.ExpectedException;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentParser;

import java.io.IOException;
import java.util.Arrays;

public class MLUploadChunkInputTest {

    MLUploadChunkInput mlUploadChunkInput;

    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();

    private byte[] content = {0x02, 0x08, 0x16, 0x0, 0x00, 0x33, (byte) 0xC6, 0x1B};

    private final String name = "test_model";

    private final Integer version = 1;

    private final Integer chunkNumber = 0;

    private final Integer totalChunks = 1;

    @Test
    public void testConstructorSuccess() throws AssertionError {
        mlUploadChunkInput = new MLUploadChunkInput(name, version, content, chunkNumber, totalChunks);
        Assert.assertNotNull(mlUploadChunkInput);
    }

    @Test
    public void testIOStreamConstructorSuccess() throws IOException {
        mlUploadChunkInput = MLUploadChunkInput.builder()
                .name(name)
                .version(version)
                .chunkNumber(chunkNumber)
                .totalChunks(totalChunks)
                .content(content)
                .build();
        BytesStreamOutput bytesStreamOutput = new BytesStreamOutput();
        mlUploadChunkInput.writeTo(bytesStreamOutput);
        MLUploadChunkInput parsedModel = new MLUploadChunkInput(bytesStreamOutput.bytes().streamInput());
        Assert.assertEquals(mlUploadChunkInput.getName(), parsedModel.getName());
        Assert.assertEquals(mlUploadChunkInput.getVersion(), parsedModel.getVersion());
        Assert.assertEquals(mlUploadChunkInput.getChunkNumber(), parsedModel.getChunkNumber());
        Assert.assertEquals(mlUploadChunkInput.getTotalChunks(), parsedModel.getTotalChunks());
        Assert.assertTrue(Arrays.equals(mlUploadChunkInput.getContent(), parsedModel.getContent()));
    }

    @Test
    public void testParseModelChunkSuccess() throws IOException {
        String query =
                String.format("{\"name\":\"%s\",\"version\":%d,\"chunk_number\":%d,\"total_chunks\":%d}", name, version, chunkNumber, totalChunks);
        XContentParser parser = parser(query);
        mlUploadChunkInput = MLUploadChunkInput.parse(parser, content);
        String actualName = mlUploadChunkInput.getName();
        Integer actualVersion = mlUploadChunkInput.getVersion();
        Integer actualChunkNumber = mlUploadChunkInput.getChunkNumber();
        Integer actualTotalChunks = mlUploadChunkInput.getTotalChunks();
        byte[] actualContent = mlUploadChunkInput.getContent();
        Assert.assertEquals(name, actualName);
        Assert.assertEquals(version, actualVersion);
        Assert.assertEquals(chunkNumber, actualChunkNumber);
        Assert.assertEquals(totalChunks, actualTotalChunks);
        Assert.assertTrue(Arrays.equals(content, actualContent));
    }



}
