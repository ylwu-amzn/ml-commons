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

public class MLUploadChunkInputTest {

    MLUploadChunkInput mlUploadChunkInput;

    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();

    private final byte[] content = new byte[] { (byte)0xe0, 0x4f, (byte)0xd0,
            0x20, (byte)0xea, 0x3a, 0x69, 0x10, (byte)0xa2, (byte)0xd8, 0x08, 0x00, 0x2b,
            0x30, 0x30, (byte)0x9d };

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
                .url(content)
                .build();
        BytesStreamOutput bytesStreamOutput = new BytesStreamOutput();
        mlUploadChunkInput.writeTo(bytesStreamOutput);
        MLUploadChunkInput parsedModel = new MLUploadChunkInput(bytesStreamOutput.bytes().streamInput());
        Assert.assertEquals(parsedModel.getName(), name);
        Assert.assertEquals(parsedModel.getVersion(), version);
        Assert.assertEquals(parsedModel.getChunkNumber(), chunkNumber);
        Assert.assertEquals(parsedModel.getTotalChunks(), totalChunks);
        Assert.assertEquals(parsedModel.getUrl(), content);
    }

    @Test
    public void testParseModelChunkSuccess() throws IOException {
        String query =
                String.format("{\"name\":%s,\"version\":%d,\"chunk_number\":%d,\"total_chunks\":%d}", name, version, chunkNumber, totalChunks);
        XContentParser parser = parser(query);
        mlUploadChunkInput = MLUploadChunkInput.parse(parser, content);
        String actualName = mlUploadChunkInput.getName();
        Integer actualVersion = mlUploadChunkInput.getVersion();
        Integer actualChunkNumber = mlUploadChunkInput.getChunkNumber();
        Integer actualTotalChunks = mlUploadChunkInput.getTotalChunks();
        Assert.assertEquals(name, actualName);
        Assert.assertEquals(version, actualVersion);
        Assert.assertEquals(chunkNumber, actualChunkNumber);
        Assert.assertEquals(totalChunks, actualTotalChunks);
    }



}
