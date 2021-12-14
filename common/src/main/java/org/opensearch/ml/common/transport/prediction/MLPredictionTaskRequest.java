/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 *
 */

package org.opensearch.ml.common.transport.prediction;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.common.io.stream.InputStreamStreamInput;
import org.opensearch.common.io.stream.OutputStreamStreamOutput;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.ml.common.MLCommonsClassLoader;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.input.Input;

import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;
import lombok.experimental.FieldDefaults;

import static org.opensearch.action.ValidateActions.addValidationError;

@Getter
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
@ToString
public class MLPredictionTaskRequest extends ActionRequest {

    String modelId;
    Input input;

    @Builder
    public MLPredictionTaskRequest(String modelId, Input input) {
        this.input = input;
        this.modelId = modelId;
    }

    public MLPredictionTaskRequest(StreamInput in) throws IOException {
        super(in);
        FunctionName functionName = in.readEnum(FunctionName.class);
        this.modelId = in.readOptionalString();
        List<Object> params = new ArrayList<>();
        params.add(functionName);
        params.add(in);
        this.input = MLCommonsClassLoader.initFunctionInput(functionName, params, null);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeOptionalString(this.modelId);
        this.input.writeTo(out);
    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException exception = null;
        if (this.input == null) {
            exception = addValidationError("ML input can't be null", exception);
        } else if (input.getFunctionName() == null) {
            exception = addValidationError("ML function name can't be null", exception);
        }

        return exception;
    }


    public static MLPredictionTaskRequest fromActionRequest(ActionRequest actionRequest) {
        if (actionRequest instanceof MLPredictionTaskRequest) {
            return (MLPredictionTaskRequest) actionRequest;
        }

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             OutputStreamStreamOutput osso = new OutputStreamStreamOutput(baos)) {
            actionRequest.writeTo(osso);
            try (StreamInput input = new InputStreamStreamInput(new ByteArrayInputStream(baos.toByteArray()))) {
                return new MLPredictionTaskRequest(input);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("failed to parse ActionRequest into MLPredictionTaskRequest", e);
        }

    }
}
