/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.model;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;
import org.opensearch.action.ActionResponse;
import org.opensearch.common.io.stream.InputStreamStreamInput;
import org.opensearch.common.io.stream.OutputStreamStreamOutput;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;

@Getter
@ToString
public class MLModelUpdateResponse extends ActionResponse implements ToXContentObject {

    String modelId;
    String status;

    @Builder
    public MLModelUpdateResponse(String modelId, String status) {
        this.modelId = modelId;
        this.status = status;
    }


    public MLModelUpdateResponse(StreamInput in) throws IOException {
        super(in);
        modelId = in.readOptionalString();
        status = in.readOptionalString();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException{
        out.writeOptionalString(modelId);
        out.writeOptionalString(status);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder xContentBuilder, Params params) throws IOException {
        xContentBuilder.startObject();
        xContentBuilder.field("model_id", modelId);
        xContentBuilder.field("status", status);
        xContentBuilder.endObject();
        return xContentBuilder;
    }

    public static MLModelUpdateResponse fromActionResponse(ActionResponse actionResponse) {
        if (actionResponse instanceof MLModelUpdateResponse) {
            return (MLModelUpdateResponse) actionResponse;
        }

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             OutputStreamStreamOutput osso = new OutputStreamStreamOutput(baos)) {
            actionResponse.writeTo(osso);
            try (StreamInput input = new InputStreamStreamInput(new ByteArrayInputStream(baos.toByteArray()))) {
                return new MLModelUpdateResponse(input);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("failed to parse ActionResponse into MLModelUpdateResponse", e);
        }
    }
}
