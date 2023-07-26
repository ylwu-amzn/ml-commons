package org.opensearch.ml.common;

import org.junit.Before;
import org.junit.Test;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;

import java.io.IOException;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class ToolMetadataTests {
    ToolMetadata toolMetadata;

    Function<XContentParser, ToolMetadata> function;

    @Before
    public void setUp() {
        toolMetadata = ToolMetadata.builder()
                .name("SearchWikipediaTool")
                .description("Useful when you need to use this tool to search general knowledge on wikipedia.")
                .build();
        function = parser -> {
            try {
                return ToolMetadata.parse(parser);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        };
    }

    @Test
    public void toXContent() throws IOException {
        XContentBuilder builder = XContentBuilder.builder(XContentType.JSON.xContent());
        toolMetadata.toXContent(builder, ToXContent.EMPTY_PARAMS);
        String toolMetadataString = TestHelper.xContentBuilderToString(builder);
        assertEquals("{\"name\":\"SearchWikipediaTool\",\"description\":\"Useful when you need to use this tool to search general knowledge on wikipedia.\"}", toolMetadataString);
    }

    @Test
    public void toXContent_nullValue() throws IOException {
        ToolMetadata emptyToolMetadata = ToolMetadata.builder().build();
        XContentBuilder builder = XContentBuilder.builder(XContentType.JSON.xContent());
        emptyToolMetadata.toXContent(builder, ToXContent.EMPTY_PARAMS);
        String toolMetadataString = TestHelper.xContentBuilderToString(builder);
        assertEquals("{}", toolMetadataString);
    }


    @Test
    public void readInputStream_Success() throws IOException {
        readInputStream(toolMetadata);
    }

    private void readInputStream(ToolMetadata toolMetadata) throws IOException {
        BytesStreamOutput bytesStreamOutput = new BytesStreamOutput();
        toolMetadata.writeTo(bytesStreamOutput);

        StreamInput streamInput = bytesStreamOutput.bytes().streamInput();
        ToolMetadata parsedToolMetadata = new ToolMetadata(streamInput);
        assertEquals(toolMetadata.getName(), parsedToolMetadata.getName());
        assertEquals(toolMetadata.getDescription(), parsedToolMetadata.getDescription());
    }
}
