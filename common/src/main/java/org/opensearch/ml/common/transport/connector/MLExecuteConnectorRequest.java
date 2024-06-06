/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.connector;

import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import lombok.experimental.FieldDefaults;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.commons.authuser.User;
import org.opensearch.core.common.io.stream.InputStreamStreamInput;
import org.opensearch.core.common.io.stream.OutputStreamStreamOutput;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.transport.MLTaskRequest;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;

import static org.opensearch.action.ValidateActions.addValidationError;

@Getter
@FieldDefaults(level = AccessLevel.PRIVATE)
@ToString
public class MLExecuteConnectorRequest extends MLTaskRequest {

    String connectorId;
    String connectorAction;
    MLInput mlInput;

    @Builder
    public MLExecuteConnectorRequest(String connectorId, String connectorAction, MLInput mlInput, boolean dispatchTask) {
        super(dispatchTask);
        this.mlInput = mlInput;
        this.connectorAction = connectorAction == null ? "predict" : connectorAction;
        this.connectorId = connectorId;
    }

    public MLExecuteConnectorRequest(String connectorId, String connectorAction, MLInput mlInput) {
        this(connectorId, connectorAction, mlInput, true);
    }

    public MLExecuteConnectorRequest(StreamInput in) throws IOException {
        super(in);
        this.connectorId = in.readString();
        this.connectorAction = in.readString();
        this.mlInput = new MLInput(in);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(this.connectorId);
        out.writeString(this.connectorAction);
        this.mlInput.writeTo(out);
    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException exception = null;
        if (this.mlInput == null) {
            exception = addValidationError("ML input can't be null", exception);
        } else if (this.mlInput.getInputDataset() == null) {
            exception = addValidationError("input data can't be null", exception);
        }

        return exception;
    }


    public static MLExecuteConnectorRequest fromActionRequest(ActionRequest actionRequest) {
        if (actionRequest instanceof MLExecuteConnectorRequest) {
            return (MLExecuteConnectorRequest) actionRequest;
        }

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             OutputStreamStreamOutput osso = new OutputStreamStreamOutput(baos)) {
            actionRequest.writeTo(osso);
            try (StreamInput input = new InputStreamStreamInput(new ByteArrayInputStream(baos.toByteArray()))) {
                return new MLExecuteConnectorRequest(input);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("failed to parse ActionRequest into MLPredictionTaskRequest", e);
        }

    }
}
