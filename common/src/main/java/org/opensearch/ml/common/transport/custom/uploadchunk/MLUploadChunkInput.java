/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom.uploadchunk;

import lombok.Builder;
import lombok.Data;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.common.xcontent.ToXContentObject;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;

import java.io.IOException;
import java.util.Objects;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;

/**
 * ML input data: algorithm name, parameters and input data set.
 */
@Data
public class MLUploadChunkInput implements ToXContentObject, Writeable {

    public static final String ALGORITHM_FIELD = "algorithm";
    public static final String NAME_FIELD = "name";
    public static final String VERSION_FIELD = "version";
    public static final String CONTENT_FIELD = "url";
    public static final String CHUNK_NUMBER_FIELD = "chunk_number";
    public static final String TOTAL_CHUNKS_FIELD = "total_chunks";

    // Algorithm name
    private FunctionName algorithm = FunctionName.CUSTOM;

    private String name;
    private Integer version;
    private byte[] content;
    private Integer chunkNumber;
    private Integer totalChunks;

    @Builder(toBuilder = true)
    public MLUploadChunkInput(String name, Integer version, byte[] url, Integer chunkNumber, Integer totalChunks) {
        Objects.requireNonNull(name);
        Objects.requireNonNull(version);
        Objects.requireNonNull(url);
        Objects.requireNonNull(chunkNumber);
        Objects.requireNonNull(totalChunks);
        this.name = name;
        this.version = version;
        this.content = url;
        this.chunkNumber = chunkNumber;
        this.algorithm = FunctionName.CUSTOM;
        this.totalChunks = totalChunks;
    }


    public MLUploadChunkInput(StreamInput in) throws IOException {
        this.algorithm = in.readEnum(FunctionName.class);
        this.name = in.readString();
        this.version = in.readInt();
        this.chunkNumber = in.readInt();
        this.totalChunks = in.readInt();
        boolean uploadModel = in.readBoolean();
        if (uploadModel) {
            this.content = in.readByteArray();
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeEnum(algorithm);
        out.writeString(name);
        out.writeInt(version);
        out.writeInt(chunkNumber);
        out.writeInt(totalChunks);
        if (content == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            out.writeByteArray(content);
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(ALGORITHM_FIELD, algorithm.name());
        builder.field(NAME_FIELD, name);
        builder.field(VERSION_FIELD, version);
        builder.field(CHUNK_NUMBER_FIELD, chunkNumber);
        builder.field(TOTAL_CHUNKS_FIELD, totalChunks);
        builder.field(CONTENT_FIELD, content);
        builder.endObject();
        return builder;
    }

    public static MLUploadChunkInput parse(XContentParser parser, byte[] content) throws IOException {
        String name = null;
        Integer version = null;
        Integer chunkNumber = null;
        Integer totalChunks = null;

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case NAME_FIELD:
                    name = parser.text();
                    break;
                case VERSION_FIELD:
                    version = parser.intValue();
                    break;
                case CHUNK_NUMBER_FIELD:
                    chunkNumber = parser.intValue();
                    break;
                case TOTAL_CHUNKS_FIELD:
                    totalChunks = parser.intValue();
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return new MLUploadChunkInput(name, version, content, chunkNumber, totalChunks);
    }


    public FunctionName getFunctionName() {
        return this.algorithm;
    }

}
