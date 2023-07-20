package org.opensearch.ml.common.transport.tools;

import org.junit.Before;
import org.junit.Test;
import org.opensearch.common.Strings;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.ml.common.ToolMetadata;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class MLToolsGetResponseTests {
    List<ToolMetadata> toolMetadataList;

    @Before
    public void setUp() {
        toolMetadataList = new ArrayList<>();
        ToolMetadata searchWikipediaTool = ToolMetadata.builder()
                .name("SearchWikipediaTool")
                .description("Useful when you need to use this tool to search general knowledge on wikipedia.")
                .build();
        toolMetadataList.add(searchWikipediaTool);
    }

    @Test
    public void writeTo_success() throws IOException {
        BytesStreamOutput bytesStreamOutput = new BytesStreamOutput();
        MLToolsGetResponse response = MLToolsGetResponse.builder().toolMetadata(toolMetadataList).build();
        response.writeTo(bytesStreamOutput);
        MLToolsGetResponse parsedResponse = new MLToolsGetResponse(bytesStreamOutput.bytes().streamInput());
        assertNotEquals(response.toolMetadataList, parsedResponse.toolMetadataList);
        assertEquals(response.toolMetadataList.get(0).getName(), parsedResponse.toolMetadataList.get(0).getName());
        assertEquals(response.toolMetadataList.get(0).getDescription(), parsedResponse.toolMetadataList.get(0).getDescription());
    }

    @Test
    public void toXContentTest() throws IOException {
        MLToolsGetResponse mlToolsGetResponse = MLToolsGetResponse.builder().toolMetadata(toolMetadataList).build();
        XContentBuilder builder = XContentFactory.contentBuilder(XContentType.JSON);
        mlToolsGetResponse.toXContent(builder, ToXContent.EMPTY_PARAMS);
        assertNotNull(builder);
        String jsonStr = Strings.toString(builder);
        assertEquals("{\"SearchWikipediaTool\":\"Useful when you need to use this tool to search general knowledge on wikipedia.\"}", jsonStr);
    }
}
