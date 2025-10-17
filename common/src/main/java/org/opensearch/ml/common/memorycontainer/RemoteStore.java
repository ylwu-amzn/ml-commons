/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.memorycontainer;

import static org.opensearch.core.xcontent.XContentParserUtils.ensureExpectedToken;

import java.io.IOException;

import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.common.io.stream.Writeable;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;

/**
 * Remote store configuration for storing memory in remote locations like AWS OpenSearch Serverless
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@EqualsAndHashCode
public class RemoteStore implements ToXContentObject, Writeable {
    
    public static final String TYPE_FIELD = "type";
    public static final String CONNECTOR_ID_FIELD = "connector_id";
    
    private String type;
    private String connectorId;
    
    public RemoteStore(StreamInput input) throws IOException {
        this.type = input.readOptionalString();
        this.connectorId = input.readOptionalString();
    }
    
    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeOptionalString(type);
        out.writeOptionalString(connectorId);
    }
    
    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        if (type != null) {
            builder.field(TYPE_FIELD, type);
        }
        if (connectorId != null) {
            builder.field(CONNECTOR_ID_FIELD, connectorId);
        }
        builder.endObject();
        return builder;
    }
    
    public static RemoteStore parse(XContentParser parser) throws IOException {
        String type = null;
        String connectorId = null;
        
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();
            
            switch (fieldName) {
                case TYPE_FIELD:
                    type = parser.text();
                    break;
                case CONNECTOR_ID_FIELD:
                    connectorId = parser.text();
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        
        return RemoteStore.builder()
            .type(type)
            .connectorId(connectorId)
            .build();
    }
}
