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

package org.opensearch.ml.common.transport.execute;

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
import org.opensearch.ml.common.MLCommonsClassLoader;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.input.Input;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

import static org.opensearch.action.ValidateActions.addValidationError;

@Getter
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
@ToString
public class MLExecuteTaskRequest extends ActionRequest {

    Input input;

    @Builder
    public MLExecuteTaskRequest(Input input) {
        this.input = input;
    }

    public MLExecuteTaskRequest(StreamInput in) throws IOException {
        super(in);
        FunctionName functionName = in.readEnum(FunctionName.class);
        List<Object> params = new ArrayList<>();
        params.add(functionName);
        params.add(in);
        this.input = MLCommonsClassLoader.initFunctionInput(functionName, params, null);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        this.input.writeTo(out);
    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException exception = null;
        if(this.input == null) {
            exception = addValidationError("ML input can't be null", exception);
        }
        if(this.input.getFunctionName() == null) {
            exception = addValidationError("function name can't be null or empty", exception);
        }

        return exception;
    }


    public static MLExecuteTaskRequest fromActionRequest(ActionRequest actionRequest) {
        if (actionRequest instanceof MLExecuteTaskRequest) {
            return (MLExecuteTaskRequest) actionRequest;
        }

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             OutputStreamStreamOutput osso = new OutputStreamStreamOutput(baos)) {
            actionRequest.writeTo(osso);
            try (StreamInput input = new InputStreamStreamInput(new ByteArrayInputStream(baos.toByteArray()))) {
                return new MLExecuteTaskRequest(input);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("failed to parse ActionRequest into MLPredictionTaskRequest", e);
        }

    }
}
