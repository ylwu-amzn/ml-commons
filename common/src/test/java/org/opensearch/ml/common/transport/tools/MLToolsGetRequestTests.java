package org.opensearch.ml.common.transport.tools;

import org.junit.Before;
import org.junit.Test;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.ml.common.ToolMetadata;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;

public class MLToolsGetRequestTests {
    private List<ToolMetadata> externalTools;

    @Before
    public void setUp() {
        externalTools = new ArrayList<>();
        ToolMetadata wikipediaTool = ToolMetadata.builder()
                .name("WikipediaTool")
                .description("Use this tool to search general knowledge on wikipedia.")
                .build();
        externalTools.add(wikipediaTool);
    }

    @Test
    public void writeTo_success() throws IOException {

        MLToolsGetRequest mlToolsGetRequest = MLToolsGetRequest.builder()
                .externalTools(externalTools)
                .build();
        BytesStreamOutput bytesStreamOutput = new BytesStreamOutput();
        mlToolsGetRequest.writeTo(bytesStreamOutput);
        MLToolsGetRequest parsedToolMetadata = new MLToolsGetRequest(bytesStreamOutput.bytes().streamInput());
        assertEquals(parsedToolMetadata.getExternalTools().get(0).getName(), externalTools.get(0).getName());
        assertEquals(parsedToolMetadata.getExternalTools().get(0).getDescription(), externalTools.get(0).getDescription());
    }

    @Test
    public void fromActionRequest_success() {
        MLToolsGetRequest mlToolsGetRequest = MLToolsGetRequest.builder().externalTools(externalTools).build();
        ActionRequest actionRequest = new ActionRequest() {
            @Override
            public ActionRequestValidationException validate() {
                return null;
            }

            @Override
            public void writeTo(StreamOutput out) throws IOException {
                mlToolsGetRequest.writeTo(out);
            }
        };
        MLToolsGetRequest result = MLToolsGetRequest.fromActionRequest(actionRequest);
        assertNotSame(result, mlToolsGetRequest);
        assertEquals(result.getExternalTools().get(0).getName(), mlToolsGetRequest.getExternalTools().get(0).getName());
    }

    @Test(expected = UncheckedIOException.class)
    public void fromActionRequest_IOException() {
        ActionRequest actionRequest = new ActionRequest() {
            @Override
            public ActionRequestValidationException validate() {
                return null;
            }

            @Override
            public void writeTo(StreamOutput out) throws IOException {
                throw new IOException("test");
            }
        };
        MLToolsGetRequest.fromActionRequest(actionRequest);
    }
}
