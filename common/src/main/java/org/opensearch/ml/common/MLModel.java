/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.ToXContentObject;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.commons.authuser.User;
import org.opensearch.ml.common.model.MLModelConfig;
import org.opensearch.ml.common.model.MLModelFormat;
import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.common.model.MLModelTaskType;
import org.opensearch.ml.common.model.TextEmbeddingModelConfig;

import java.io.IOException;
import java.time.Instant;
import java.util.Base64;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.common.CommonValue.USER;

@Getter
public class MLModel implements ToXContentObject {
    public static final String ALGORITHM_FIELD = "algorithm";
    public static final String MODEL_NAME_FIELD = "name";
    public static final String MODEL_VERSION_FIELD = "version";
    public static final String OLD_MODEL_CONTENT_FIELD = "content";
    public static final String MODEL_CONTENT_FIELD = "model_content";

    public static final String DESCRIPTION_FIELD = "description";
    public static final String MODEL_FORMAT_FIELD = "model_format";
    public static final String MODEL_TASK_TYPE_FIELD = "model_task_type";
    public static final String MODEL_STATE_FIELD = "model_state";
    public static final String MODEL_CONTENT_SIZE_IN_BYTES_FIELD = "model_content_size_in_bytes";
    public static final String MODEL_CONTENT_HASH_FIELD = "model_content_hash";
    public static final String MODEL_CONFIG_FIELD = "model_config";
    public static final String CREATED_TIME_FIELD = "created_time";
    public static final String LATEST_UPLOADED_TIME_FIELD = "latest_uploaded_time";//TODO: change to last, to keep consistent with MLTask?
    public static final String LATEST_LOADED_TIME_FIELD = "latest_loaded_time";
    public static final String LATEST_UNLOADED_TIME_FIELD = "latest_unloaded_time";

    public static final String MODEL_ID_FIELD = "model_id";
    public static final String CHUNK_NUMBER_FIELD = "chunk_number";
    public static final String TOTAL_CHUNKS_FIELD = "total_chunks";

    private String name;
    private FunctionName algorithm;
    private Integer version;
    private String content;
    private User user;

    @Setter
    private String description;
    private MLModelFormat modelFormat;
    private MLModelTaskType modelTaskType;
    private MLModelState modelState;
    private Long modelContentSizeInBytes;
    private String modelContentHash;
    private MLModelConfig modelConfig;
    private Instant createdTime;
    private Instant latestUploadedTime;
    private Instant latestLoadedTime;
    private Instant latestUnloadedTime;

    @Setter
    private String modelId; // model chunk doc only
    private Integer chunkNumber; // model chunk doc only
    private Integer totalChunks; // model chunk doc only

    @Builder
    public MLModel(String name, FunctionName algorithm, Integer version, String content, User user, String description, MLModelFormat modelFormat, MLModelTaskType modelTaskType, MLModelState modelState, Long modelContentSizeInBytes, String modelContentHash, MLModelConfig modelConfig, Instant createdTime, Instant latestUploadedTime, Instant latestLoadedTime, Instant latestUnloadedTime, String modelId, Integer chunkNumber, Integer totalChunks) {
        this.name = name;
        this.algorithm = algorithm;
        this.version = version;
        this.content = content;
        this.user = user;
        this.description = description;
        this.modelFormat = modelFormat;
        this.modelTaskType = modelTaskType;
        this.modelState = modelState;
        this.modelContentSizeInBytes = modelContentSizeInBytes;
        this.modelContentHash = modelContentHash;
        this.modelConfig = modelConfig;
        this.createdTime = createdTime;
        this.latestUploadedTime = latestUploadedTime;
        this.latestLoadedTime = latestLoadedTime;
        this.latestUnloadedTime = latestUnloadedTime;
        this.modelId = modelId;
        this.chunkNumber = chunkNumber;
        this.totalChunks = totalChunks;
    }

    public MLModel(FunctionName algorithm, Model model) {
        this.name = model.getName();
        this.algorithm = algorithm;
        this.version = model.getVersion();
        this.content = Base64.getEncoder().encodeToString(model.getContent());
    }

    public MLModel(StreamInput input) throws IOException{
        name = input.readOptionalString();
        algorithm = input.readEnum(FunctionName.class);
        version = input.readInt();
        content = input.readOptionalString();
        if (input.readBoolean()) {
            this.user = new User(input);
        } else {
            user = null;
        }
        if (input.available() > 0) {
            description = input.readOptionalString();
            if (input.readBoolean()) {
                modelFormat = input.readEnum(MLModelFormat.class);
            }
            if (input.readBoolean()) {
                modelTaskType = input.readEnum(MLModelTaskType.class);
            }
            if (input.readBoolean()) {
                modelState = input.readEnum(MLModelState.class);
            }
            modelContentSizeInBytes = input.readOptionalLong();
            modelContentHash = input.readOptionalString();
            if (input.readBoolean()) {
                modelConfig = new TextEmbeddingModelConfig(input);
            }
            createdTime = input.readOptionalInstant();
            latestUploadedTime = input.readOptionalInstant();
            latestLoadedTime = input.readOptionalInstant();
            latestUnloadedTime = input.readOptionalInstant();
            modelId = input.readOptionalString();
            chunkNumber = input.readOptionalInt();
            totalChunks = input.readOptionalInt();
        }
    }

    public void writeTo(StreamOutput out) throws IOException {
        out.writeOptionalString(name);
        out.writeEnum(algorithm);
        out.writeInt(version);
        out.writeOptionalString(content);
        if (user != null) {
            out.writeBoolean(true); // user exists
            user.writeTo(out);
        } else {
            out.writeBoolean(false); // user does not exist
        }
        out.writeOptionalString(description);
        if (modelFormat != null) {
            out.writeBoolean(true);
            out.writeEnum(modelFormat);
        } else {
            out.writeBoolean(false);
        }
        if (modelTaskType != null) {
            out.writeBoolean(true);
            out.writeEnum(modelTaskType);
        } else {
            out.writeBoolean(false);
        }
        if (modelState != null) {
            out.writeBoolean(true);
            out.writeEnum(modelState);
        } else {
            out.writeBoolean(false);
        }
        out.writeOptionalLong(modelContentSizeInBytes);
        out.writeOptionalString(modelContentHash);
        if (modelConfig != null) {
            out.writeBoolean(true);
            modelConfig.writeTo(out);
        } else {
            out.writeBoolean(false);
        }
        out.writeOptionalInstant(createdTime);
        out.writeOptionalInstant(latestUploadedTime);
        out.writeOptionalInstant(latestLoadedTime);
        out.writeOptionalInstant(latestUnloadedTime);
        out.writeOptionalString(modelId);
        out.writeOptionalInt(chunkNumber);
        out.writeOptionalInt(totalChunks);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        if (name != null) {
            builder.field(MODEL_NAME_FIELD, name);
        }
        if (algorithm != null) {
            builder.field(ALGORITHM_FIELD, algorithm);
        }
        if (version != null) {
            builder.field(MODEL_VERSION_FIELD, version);
        }
        if (content != null) {
            builder.field(MODEL_CONTENT_FIELD, content);
        }
        if (user != null) {
            builder.field(USER, user);
        }
        if (description != null) {
            builder.field(DESCRIPTION_FIELD, description);
        }
        if (modelFormat != null) {
            builder.field(MODEL_FORMAT_FIELD, modelFormat);
        }
        if (modelTaskType != null) {
            builder.field(MODEL_TASK_TYPE_FIELD, modelTaskType);
        }
        if (modelState != null) {
            builder.field(MODEL_STATE_FIELD, modelState);
        }
        if (modelContentSizeInBytes != null) {
            builder.field(MODEL_CONTENT_SIZE_IN_BYTES_FIELD, modelContentSizeInBytes);
        }
        if (modelContentHash != null) {
            builder.field(MODEL_CONTENT_HASH_FIELD, modelContentHash);
        }
        if (modelConfig != null) {
            builder.field(MODEL_CONFIG_FIELD, modelConfig);
        }
        if (createdTime != null) {
            builder.field(CREATED_TIME_FIELD, createdTime.toEpochMilli());
        }
        if (latestUploadedTime != null) {
            builder.field(LATEST_UPLOADED_TIME_FIELD, latestUploadedTime.toEpochMilli());
        }
        if (latestLoadedTime != null) {
            builder.field(LATEST_LOADED_TIME_FIELD, latestLoadedTime.toEpochMilli());
        }
        if (latestUnloadedTime != null) {
            builder.field(LATEST_UNLOADED_TIME_FIELD, latestUnloadedTime.toEpochMilli());
        }
        if (modelId != null) {
            builder.field(MODEL_ID_FIELD, modelId);
        }
        if (chunkNumber != null) {
            builder.field(CHUNK_NUMBER_FIELD, chunkNumber);
        }
        if (totalChunks != null) {
            builder.field(TOTAL_CHUNKS_FIELD, totalChunks);
        }
        builder.endObject();
        return builder;
    }

    public static MLModel parse(XContentParser parser) throws IOException {
        String name = null;
        FunctionName algorithm = null;
        Integer version = null;
        String content = null;
        String oldContent = null;
        User user = null;

        String description = null;;
        MLModelFormat modelFormat = null;
        MLModelTaskType modelTaskType = null;
        MLModelState modelState = null;
        Long modelContentSizeInBytes = null;
        String modelContentHash = null;
        MLModelConfig modelConfig = null;
        Instant createdTime = null;
        Instant lastUploadedTime = null;
        Instant lastLoadedTime = null;
        Instant lastUnloadedTime = null;
        String modelId = null;
        Integer chunkNumber = null;
        Integer totalChunks = null;

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case MODEL_NAME_FIELD:
                    name = parser.text();
                    break;
                case MODEL_CONTENT_FIELD:
                    content = parser.text();
                    break;
                case OLD_MODEL_CONTENT_FIELD:
                    oldContent = parser.text();
                    break;
                case MODEL_VERSION_FIELD:
                    version = parser.intValue(false);
                    break;
                case CHUNK_NUMBER_FIELD:
                    chunkNumber = parser.intValue(false);
                    break;
                case TOTAL_CHUNKS_FIELD:
                    totalChunks = parser.intValue(false);
                    break;
                case USER:
                    user = User.parse(parser);
                    break;
                case ALGORITHM_FIELD:
                    algorithm = FunctionName.from(parser.text());
                    break;
                case MODEL_ID_FIELD:
                    modelId = parser.text();
                    break;
                case DESCRIPTION_FIELD:
                    description = parser.text();
                    break;
                case MODEL_FORMAT_FIELD:
                    modelFormat = MLModelFormat.from(parser.text());
                    break;
                case MODEL_TASK_TYPE_FIELD:
                    modelTaskType = MLModelTaskType.from(parser.text());
                    break;
                case MODEL_STATE_FIELD:
                    modelState = MLModelState.from(parser.text());
                    break;
                case MODEL_CONTENT_SIZE_IN_BYTES_FIELD:
                    modelContentSizeInBytes = parser.longValue();
                    break;
                case MODEL_CONTENT_HASH_FIELD:
                    modelContentHash = parser.text();
                    break;
                case MODEL_CONFIG_FIELD:
                    modelConfig = TextEmbeddingModelConfig.parse(parser);
                    break;
                case CREATED_TIME_FIELD:
                    createdTime = Instant.ofEpochMilli(parser.longValue());
                    break;
                case LATEST_UPLOADED_TIME_FIELD:
                    lastUploadedTime = Instant.ofEpochMilli(parser.longValue());
                    break;
                case LATEST_LOADED_TIME_FIELD:
                    lastLoadedTime = Instant.ofEpochMilli(parser.longValue());
                    break;
                case LATEST_UNLOADED_TIME_FIELD:
                    lastUnloadedTime = Instant.ofEpochMilli(parser.longValue());
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return MLModel.builder()
                .name(name)
                .algorithm(algorithm)
                .version(version)
                .content(content == null ? oldContent : content)
                .user(user)
                .description(description)
                .modelFormat(modelFormat)
                .modelTaskType(modelTaskType)
                .modelState(modelState)
                .modelContentSizeInBytes(modelContentSizeInBytes)
                .modelContentHash(modelContentHash)
                .modelConfig(modelConfig)
                .createdTime(createdTime)
                .latestUploadedTime(lastUploadedTime)
                .latestLoadedTime(lastLoadedTime)
                .latestUnloadedTime(lastUnloadedTime)
                .modelId(modelId)
                .chunkNumber(chunkNumber)
                .totalChunks(totalChunks)
                .build();
        }

    public static MLModel fromStream(StreamInput in) throws IOException {
        MLModel mlModel = new MLModel(in);
        return mlModel;
    }
}
