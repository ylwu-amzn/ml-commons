/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom_model.sync;

import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;
import lombok.experimental.FieldDefaults;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.common.io.stream.InputStreamStreamInput;
import org.opensearch.common.io.stream.OutputStreamStreamOutput;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.ml.common.transport.MLTaskRequest;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;

import static org.opensearch.action.ValidateActions.addValidationError;

@Getter
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
@ToString
public class MLSyncUpRequest extends MLTaskRequest {

    private String modelId;
    private String[] addedWorkerNodes;
    private String[] removedWorkerNodes;
    boolean async;

    @Builder
    public MLSyncUpRequest(String modelId, String[] addedWorkerNodes, String[] removedWorkerNodes, boolean async, boolean dispatchTask) {
        super(dispatchTask);
        this.modelId = modelId;
        this.addedWorkerNodes = addedWorkerNodes;
        this.removedWorkerNodes = removedWorkerNodes;
        this.async = async;
    }

    public MLSyncUpRequest(String modelId, String[] addedWorkerNodes, boolean async) {
        this(modelId, addedWorkerNodes, null, async, true);
    }

    public MLSyncUpRequest(StreamInput in) throws IOException {
        super(in);
        this.modelId = in.readString();
        this.addedWorkerNodes = in.readOptionalStringArray();
        this.removedWorkerNodes = in.readOptionalStringArray();
        this.async = in.readBoolean();
    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException exception = null;
        if (modelId == null) {
            exception = addValidationError("ML model id can't be null", exception);
        }

        return exception;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(modelId);
        out.writeOptionalStringArray(addedWorkerNodes);
        out.writeOptionalStringArray(removedWorkerNodes);
        out.writeBoolean(async);
    }

    public static MLSyncUpRequest fromActionRequest(ActionRequest actionRequest) {
        if (actionRequest instanceof MLSyncUpRequest) {
            return (MLSyncUpRequest) actionRequest;
        }

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             OutputStreamStreamOutput osso = new OutputStreamStreamOutput(baos)) {
            actionRequest.writeTo(osso);
            try (StreamInput input = new InputStreamStreamInput(new ByteArrayInputStream(baos.toByteArray()))) {
                return new MLSyncUpRequest(input);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("failed to parse ActionRequest into MLSyncUpRequest", e);
        }

    }

}
