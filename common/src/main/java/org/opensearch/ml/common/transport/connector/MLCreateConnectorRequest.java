/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.connector;

import lombok.Builder;
import lombok.Getter;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.common.io.stream.InputStreamStreamInput;
import org.opensearch.common.io.stream.OutputStreamStreamOutput;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.ml.common.connector.Connector;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;

import static org.opensearch.action.ValidateActions.addValidationError;

@Getter
public class MLCreateConnectorRequest extends ActionRequest {
    private Connector connector;
    private boolean dryRun;
    private boolean addAllBackendRoles;

    @Builder
    public MLCreateConnectorRequest(Connector connector, boolean dryRun, boolean addAllBackendRoles) {
        this.connector = connector;
        this.dryRun = dryRun;
        this.addAllBackendRoles = addAllBackendRoles;
    }

    public MLCreateConnectorRequest(Connector connector) {
        this.connector = connector;
        this.dryRun = false;
        this.addAllBackendRoles = false;
    }

    public MLCreateConnectorRequest(StreamInput in) throws IOException {
        super(in);
        this.connector = Connector.fromStream(in);
        this.dryRun = in.readBoolean();
        this.addAllBackendRoles = in.readBoolean();
    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException exception = null;
        if (connector == null) {
            exception = addValidationError("ML Connector input can't be null", exception);
        }

        return exception;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        this.connector.writeTo(out);
        out.writeBoolean(dryRun);
        out.writeBoolean(addAllBackendRoles);
    }

    public static MLCreateConnectorRequest fromActionRequest(ActionRequest actionRequest) {
        if (actionRequest instanceof MLCreateConnectorRequest) {
            return (MLCreateConnectorRequest) actionRequest;
        }

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             OutputStreamStreamOutput osso = new OutputStreamStreamOutput(baos)) {
            actionRequest.writeTo(osso);
            try (StreamInput input = new InputStreamStreamInput(new ByteArrayInputStream(baos.toByteArray()))) {
                return new MLCreateConnectorRequest(input);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to parse ActionRequest into MLCreateConnectorRequest", e);
        }
    }
}
