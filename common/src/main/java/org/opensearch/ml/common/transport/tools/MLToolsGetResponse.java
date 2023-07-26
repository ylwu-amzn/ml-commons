package org.opensearch.ml.common.transport.tools;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;
import org.opensearch.action.ActionResponse;
import org.opensearch.common.io.stream.*;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.ml.common.ToolMetadata;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;

@Getter
@ToString
public class MLToolsGetResponse extends ActionResponse implements ToXContentObject {

    List<ToolMetadata> toolMetadataList;

    @Builder
    public MLToolsGetResponse(List<ToolMetadata> toolMetadata) {
        this.toolMetadataList = toolMetadata;
    }
    public MLToolsGetResponse(StreamInput in) throws IOException {
        super(in);
        this.toolMetadataList = in.readList(ToolMetadata::new);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeList(toolMetadataList);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder xContentBuilder, ToXContent.Params params) throws IOException {
        xContentBuilder.startObject();
        for (ToolMetadata toolMetadata : toolMetadataList) {
            xContentBuilder.field(toolMetadata.getName(), toolMetadata.getDescription());
        }
        xContentBuilder.endObject();
        return xContentBuilder;
    }

    public static MLToolsGetResponse fromActionResponse(ActionResponse actionResponse) {
        if (actionResponse instanceof MLToolsGetResponse) {
            return (MLToolsGetResponse) actionResponse;
        }

        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
             OutputStreamStreamOutput osso = new OutputStreamStreamOutput(baos)) {
            actionResponse.writeTo(osso);
            try (StreamInput input = new InputStreamStreamInput(new ByteArrayInputStream(baos.toByteArray()))) {
                return new MLToolsGetResponse(input);
            }
        }
        catch (IOException e) {
            throw new UncheckedIOException("failed to parse ActionResponse into MLToolsGetResponse", e);
        }
    }
}
