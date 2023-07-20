package org.opensearch.ml.common.transport.tools;

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
import org.opensearch.ml.common.ToolMetadata;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;

@Getter
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
@ToString
public class MLToolsGetRequest extends ActionRequest {

    List<ToolMetadata> externalTools;

    @Builder
    public MLToolsGetRequest(List<ToolMetadata> externalTools) {
        this.externalTools = externalTools;
    }

    public MLToolsGetRequest(StreamInput in) throws IOException {
        super(in);
        this.externalTools = in.readList(ToolMetadata::new);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeList(this.externalTools);
    }

    @Override
    public ActionRequestValidationException validate() {
        return null;
    }

    public static MLToolsGetRequest fromActionRequest(ActionRequest actionRequest) {
        if (actionRequest instanceof  MLToolsGetRequest) {
            return (MLToolsGetRequest)actionRequest;
        }

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             OutputStreamStreamOutput osso = new OutputStreamStreamOutput(baos)) {
            actionRequest.writeTo(osso);
            try (StreamInput input = new InputStreamStreamInput(new ByteArrayInputStream(baos.toByteArray()))) {
                return new MLToolsGetRequest(input);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("failed to parse ActionRequest into MLToolsGetRequest", e);
        }
    }

}
