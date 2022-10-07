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
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.transport.custom_model.upload.MLUploadInput;

import java.io.IOException;

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
    public static final String UPLOAD_INPUT_FIELD = "upload_model_input";

    private FunctionName algorithm = FunctionName.CUSTOM;

    private String taskId;
    private String modelId;
    private String workerNodeId;
    private MLForwardRequestType requestType;
    private MLTask mlTask;
    MLInput modelInput;
    private String error;
    private String[] workerNodes;
    private MLUploadInput uploadInput;

    @Builder(toBuilder = true)
    public MLForwardInput(String taskId, String modelId, String workerNodeId, MLForwardRequestType requestType,
                          MLTask mlTask, MLInput modelInput,
                          String error, String[] workerNodes, MLUploadInput uploadInput) {
        this.taskId = taskId;
        this.modelId = modelId;
        this.workerNodeId = workerNodeId;
        this.requestType = requestType;
        this.mlTask = mlTask;
        this.modelInput = modelInput;
        this.error = error;
        this.workerNodes = workerNodes;
        this.uploadInput = uploadInput;
    }

    public MLForwardInput(StreamInput in) throws IOException {
        this.algorithm = in.readEnum(FunctionName.class);
        this.taskId = in.readOptionalString();
        this.modelId = in.readOptionalString();
        this.workerNodeId = in.readOptionalString();
        this.requestType = in.readEnum(MLForwardRequestType.class);
        if (in.readBoolean()) {
            mlTask = new MLTask(in);
        }
        if (in.readBoolean()) {
            this.modelInput = new MLInput(in);
        }
        this.error = in.readOptionalString();
        this.workerNodes = in.readOptionalStringArray();
        if (in.readBoolean()) {
            this.uploadInput = new MLUploadInput(in);
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeEnum(algorithm);
        out.writeOptionalString(taskId);
        out.writeOptionalString(modelId);
        out.writeOptionalString(workerNodeId);
        out.writeEnum(requestType);
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
        out.writeOptionalString(error);
        out.writeOptionalStringArray(workerNodes);
        if (uploadInput != null) {
            out.writeBoolean(true);
            uploadInput.writeTo(out);
        } else {
            out.writeBoolean(false);
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(ALGORITHM_FIELD, algorithm.name());
        builder.field(TASK_ID_FIELD, taskId);
        builder.field(MODEL_ID_FIELD, modelId);
        builder.field(WORKER_NODE_ID_FIELD, workerNodeId);
        builder.field(REQUEST_TYPE_FIELD, requestType);
        if (mlTask != null) {
            mlTask.toXContent(builder, params);
        }
        if (error != null) {
            builder.field(ERROR_FIELD, requestType);
        }
        if (workerNodes != null) {
            builder.field(WORKER_NODES_FIELD, workerNodes);
        }
        if (uploadInput != null) {
            builder.field(UPLOAD_INPUT_FIELD, uploadInput);
        }
        builder.endObject();
        return builder;
    }

}
