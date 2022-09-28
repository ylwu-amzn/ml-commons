/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom_model.predict;

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
public class MLPredictModelRequest extends MLTaskRequest {

    TextEmbeddingModelInput modelInput;
    boolean async;

    @Builder
    public MLPredictModelRequest(TextEmbeddingModelInput modelInput, boolean async, boolean dispatchTask) {
        super(dispatchTask);
        this.modelInput = modelInput;
        this.async = async;
    }

    public MLPredictModelRequest(TextEmbeddingModelInput modelInput, boolean async) {
        this(modelInput, async, true);
    }

    public MLPredictModelRequest(StreamInput in) throws IOException {
        super(in);
        this.modelInput = new TextEmbeddingModelInput(in);
        this.async = in.readBoolean();
    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException exception = null;
        if (modelInput == null) {
            exception = addValidationError("ML input can't be null", exception);
        }

        return exception;
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        this.modelInput.writeTo(out);
        out.writeBoolean(async);
    }

    public static MLPredictModelRequest fromActionRequest(ActionRequest actionRequest) {
        if (actionRequest instanceof MLPredictModelRequest) {
            return (MLPredictModelRequest) actionRequest;
        }

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             OutputStreamStreamOutput osso = new OutputStreamStreamOutput(baos)) {
            actionRequest.writeTo(osso);
            try (StreamInput input = new InputStreamStreamInput(new ByteArrayInputStream(baos.toByteArray()))) {
                return new MLPredictModelRequest(input);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to parse ActionRequest into MLPredictModelRequest", e);
        }

    }
}
