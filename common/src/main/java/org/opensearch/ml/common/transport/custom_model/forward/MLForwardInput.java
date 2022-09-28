/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom_model.forward;

import lombok.Builder;
import lombok.Data;
import lombok.extern.log4j.Log4j2;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.common.xcontent.ToXContentObject;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.model.MLModelConfig;
import org.opensearch.ml.common.model.MLModelFormat;
import org.opensearch.ml.common.model.TextEmbeddingModelConfig;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;

@Data
@Log4j2
public class MLForwardInput implements ToXContentObject, Writeable {

    public static final String ALGORITHM_FIELD = "algorithm";
    public static final String NAME_FIELD = "name";
    public static final String VERSION_FIELD = "version";
    public static final String TASK_ID_FIELD = "task_id";
    public static final String MODEL_ID_FIELD = "model_id";
    public static final String WORKER_NODE_ID_FIELD = "worker_node_id";
    public static final String REQUEST_TYPE_FIELD = "request_type";
    public static final String ML_TASK_FIELD = "ml_task";
    public static final String URL_FIELD = "url";
    public static final String PREDICT_INPUT_FIELD = "predict_input";
    public static final String ERROR_FIELD = "error";
    public static final String WORKER_NODES_FIELD = "worker_nodes";

    private FunctionName algorithm = FunctionName.CUSTOM;

    private String name;
    private Integer version;
    private String taskId;
    private String modelId;
    private String workerNodeId;
    private MLForwardRequestType requestType;
    private MLTask mlTask;
    private String url;
    MLInput modelInput;
    private MLModelFormat modelFormat;
    private MLModelConfig modelConfig;
    private String error;
    private String[] workerNodes;

    @Builder(toBuilder = true)
    public MLForwardInput(String name, Integer version, String taskId,String modelId, String workerNodeId, MLForwardRequestType requestType,
                          MLTask mlTask, String url, MLInput modelInput,
                          MLModelFormat modelFormat, MLModelConfig modelConfig,
                          String error, String[] workerNodes) {
        this.name = name;
        this.version = version;
        this.taskId = taskId;
        this.modelId = modelId;
        this.workerNodeId = workerNodeId;
        this.requestType = requestType;
        this.mlTask = mlTask;
        this.url = url;
        this.modelInput = modelInput;
        this.modelFormat = modelFormat;
        this.modelConfig = modelConfig;
        this.error = error;
        this.workerNodes = workerNodes;
    }

    public MLForwardInput(StreamInput in) throws IOException {
        this.name = in.readOptionalString();
        this.version = in.readOptionalInt();
        this.algorithm = in.readEnum(FunctionName.class);
        this.taskId = in.readOptionalString();
        this.modelId = in.readOptionalString();
        this.workerNodeId = in.readOptionalString();
        this.requestType = in.readEnum(MLForwardRequestType.class);
        this.url = in.readOptionalString();
        if (in.readBoolean()) {
            mlTask = new MLTask(in);
        }
        if (in.readBoolean()) {
            this.modelInput = new MLInput(in);
        }
        if (in.readBoolean()) {
            this.modelFormat = in.readEnum(MLModelFormat.class);
        }
        if (in.readBoolean()) {
            this.modelConfig = new TextEmbeddingModelConfig(in);
        }
        this.error = in.readOptionalString();
        this.workerNodes = in.readOptionalStringArray();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeOptionalString(name);
        out.writeOptionalInt(version);
        out.writeEnum(algorithm);
        out.writeOptionalString(taskId);
        out.writeOptionalString(modelId);
        out.writeOptionalString(workerNodeId);
        out.writeEnum(requestType);
        out.writeOptionalString(url);
        if (this.mlTask != null) {
            out.writeBoolean(true);
            mlTask.writeTo(out);
        } else {
            out.writeBoolean(false);
        }
        if (modelInput != null) {
            out.writeBoolean(true);
            modelInput.writeTo(out);
        } else {
            out.writeBoolean(false);
        }
        if (modelFormat != null) {
            out.writeBoolean(true);
            out.writeEnum(modelFormat);
        } else {
            out.writeBoolean(false);
        }
        if (modelConfig != null) {
            out.writeBoolean(true);
            modelConfig.writeTo(out);
        } else {
            out.writeBoolean(false);
        }
        out.writeOptionalString(error);
        out.writeOptionalStringArray(workerNodes);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(ALGORITHM_FIELD, algorithm.name());
        builder.field(NAME_FIELD, name);
        builder.field(VERSION_FIELD, version);
        builder.field(TASK_ID_FIELD, taskId);
        builder.field(MODEL_ID_FIELD, modelId);
        builder.field(WORKER_NODE_ID_FIELD, workerNodeId);
        builder.field(REQUEST_TYPE_FIELD, requestType);
        if (mlTask != null) {
            mlTask.toXContent(builder, params);
        }
        if (url != null) {
            builder.field(URL_FIELD, url);
        }
        if (modelFormat != null) {
            builder.field(MLModel.MODEL_FORMAT_FIELD, modelFormat);
        }
        if (modelConfig != null) {
            builder.field(MLModel.MODEL_CONFIG_FIELD, modelConfig);
        }
        if (error != null) {
            builder.field(ERROR_FIELD, requestType);
        }
        if (workerNodes != null) {
            builder.field(WORKER_NODES_FIELD, workerNodes);
        }
        builder.endObject();
        return builder;
    }

    public static MLForwardInput parse(XContentParser parser) throws IOException {
        String algorithmName = null;
        String name = null;
        Integer version = null;
        String taskId = null;
        String modelId = null;
        String workerNodeId = null;
        MLForwardRequestType requestType = null;
        MLTask mlTask = null;
        String url = null;
        MLModelFormat modelFormat = null;
        MLModelConfig modelConfig = null;
        String error = null;
        List<String> workerNodes = null;

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case ALGORITHM_FIELD:
                    algorithmName = parser.text().toUpperCase(Locale.ROOT);
                    break;
                case NAME_FIELD:
                    name = parser.text();
                    break;
                case VERSION_FIELD:
                    version = parser.intValue();
                    break;
                case TASK_ID_FIELD:
                    taskId = parser.text();
                    break;
                case MODEL_ID_FIELD:
                    modelId = parser.text();
                    break;
                case WORKER_NODE_ID_FIELD:
                    workerNodeId = parser.text();
                    break;
                case REQUEST_TYPE_FIELD:
                    requestType = MLForwardRequestType.valueOf(parser.text().toUpperCase(Locale.ROOT));
                    break;
                case ML_TASK_FIELD:
                    mlTask = MLTask.parse(parser);
                    break;
                case URL_FIELD:
                    url = parser.text();
                    break;
                case MLModel.MODEL_FORMAT_FIELD:
                    modelFormat = MLModelFormat.from(parser.text().toUpperCase(Locale.ROOT));
                    break;
                case MLModel.MODEL_CONFIG_FIELD:
                    modelConfig = TextEmbeddingModelConfig.parse(parser);
                    break;
                case ERROR_FIELD:
                    error = parser.text();
                    break;
                case WORKER_NODES_FIELD:
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return new MLForwardInput(name, version, taskId, modelId, workerNodeId, requestType, mlTask, url, null, modelFormat, modelConfig, error, workerNodes.toArray(new String[0]));
    }


    public FunctionName getFunctionName() {
        return this.algorithm;
    }

}
