/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.model;

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
import org.opensearch.ml.common.transport.register.MLUpdateModelInput;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;

import static org.opensearch.action.ValidateActions.addValidationError;

@Getter
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
@ToString
public class MLModelUpdateRequest extends ActionRequest {

    String modelId;
    MLUpdateModelInput updateModelInput;

    @Builder
    public MLModelUpdateRequest(String modelId, MLUpdateModelInput updateModelInput) {
        this.modelId = modelId;
        this.updateModelInput = updateModelInput;
    }

    public MLModelUpdateRequest(StreamInput in) throws IOException {
        super(in);
        this.modelId = in.readString();
        this.updateModelInput = new MLUpdateModelInput(in);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(this.modelId);
        this.updateModelInput.writeTo(out);
    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException exception = null;

        if (this.modelId == null) {
            exception = addValidationError("ML model id can't be null", exception);
        }

        if (this.updateModelInput == null) {
            exception = addValidationError("No updated content provided", exception);
        }

        return exception;
    }

    public static MLModelUpdateRequest fromActionRequest(ActionRequest actionRequest) {
        if (actionRequest instanceof MLModelUpdateRequest) {
            return (MLModelUpdateRequest)actionRequest;
        }

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             OutputStreamStreamOutput osso = new OutputStreamStreamOutput(baos)) {
            actionRequest.writeTo(osso);
            try (StreamInput input = new InputStreamStreamInput(new ByteArrayInputStream(baos.toByteArray()))) {
                return new MLModelUpdateRequest(input);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("failed to parse ActionRequest into MLModelUpdateRequest", e);
        }
    }
}
