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
    MLInput mlInput;
    @Setter
    User user;

    @Builder
    public MLExecuteConnectorRequest(String connectorId, MLInput mlInput, boolean dispatchTask, User user) {
        super(dispatchTask);
        this.mlInput = mlInput;
        this.connectorId = connectorId;
        this.user = user;
    }

    public MLExecuteConnectorRequest(String connectorId, MLInput mlInput) {
        this(connectorId, mlInput, true, null);
    }

    public MLExecuteConnectorRequest(String connectorId, MLInput mlInput, User user) {
        this(connectorId, mlInput, true, user);
    }

    public MLExecuteConnectorRequest(StreamInput in) throws IOException {
        super(in);
        this.connectorId = in.readOptionalString();
        this.mlInput = new MLInput(in);
        if (in.readBoolean()) {
            this.user = new User(in);
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeOptionalString(this.connectorId);
        this.mlInput.writeTo(out);
        if (user != null) {
            out.writeBoolean(true);
            user.writeTo(out);
        } else {
            out.writeBoolean(false);
        }
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
