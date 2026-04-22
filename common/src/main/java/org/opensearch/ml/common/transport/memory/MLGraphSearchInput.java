/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.memory;

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
import lombok.Getter;

/**
 * Input for graph search operations
 */
@Getter
@Builder
@AllArgsConstructor
public class MLGraphSearchInput implements ToXContentObject, Writeable {

    public static final String MEMORY_CONTAINER_ID_FIELD = "memory_container_id";
    public static final String QUERY_FIELD = "query";
    public static final String TOP_K_FIELD = "top_k";
    public static final String ENTITY_ID_FIELD = "entity_id";
    public static final String MAX_DEPTH_FIELD = "max_depth";
    public static final String SEARCH_TYPE_FIELD = "search_type";
    public static final String ENTITY_TYPE_FIELD = "entity_type";

    // Required fields
    private String memoryContainerId;
    private String query;

    // Optional fields
    @Builder.Default
    private Integer topK = 10;
    private String entityId; // For relationship traversal
    @Builder.Default
    private Integer maxDepth = 2;
    @Builder.Default
    private String searchType = "hybrid"; // "text", "traversal", "hybrid"
    private String entityType; // Filter by entity type

    public MLGraphSearchInput(StreamInput input) throws IOException {
        this.memoryContainerId = input.readString();
        this.query = input.readString();
        this.topK = input.readOptionalVInt();
        this.entityId = input.readOptionalString();
        this.maxDepth = input.readOptionalVInt();
        this.searchType = input.readOptionalString();
        this.entityType = input.readOptionalString();
    }

    public static MLGraphSearchInput parse(XContentParser parser) throws IOException {
        String memoryContainerId = null;
        String query = null;
        Integer topK = 10;
        String entityId = null;
        Integer maxDepth = 2;
        String searchType = "hybrid";
        String entityType = null;

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case MEMORY_CONTAINER_ID_FIELD:
                    memoryContainerId = parser.text();
                    break;
                case QUERY_FIELD:
                    query = parser.text();
                    break;
                case TOP_K_FIELD:
                    topK = parser.intValue();
                    break;
                case ENTITY_ID_FIELD:
                    entityId = parser.text();
                    break;
                case MAX_DEPTH_FIELD:
                    maxDepth = parser.intValue();
                    break;
                case SEARCH_TYPE_FIELD:
                    searchType = parser.text();
                    break;
                case ENTITY_TYPE_FIELD:
                    entityType = parser.text();
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }

        return MLGraphSearchInput.builder()
            .memoryContainerId(memoryContainerId)
            .query(query)
            .topK(topK)
            .entityId(entityId)
            .maxDepth(maxDepth)
            .searchType(searchType)
            .entityType(entityType)
            .build();
    }

    @Override
    public void writeTo(StreamOutput output) throws IOException {
        output.writeString(memoryContainerId);
        output.writeString(query);
        output.writeOptionalVInt(topK);
        output.writeOptionalString(entityId);
        output.writeOptionalVInt(maxDepth);
        output.writeOptionalString(searchType);
        output.writeOptionalString(entityType);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(MEMORY_CONTAINER_ID_FIELD, memoryContainerId);
        builder.field(QUERY_FIELD, query);
        if (topK != null) {
            builder.field(TOP_K_FIELD, topK);
        }
        if (entityId != null) {
            builder.field(ENTITY_ID_FIELD, entityId);
        }
        if (maxDepth != null) {
            builder.field(MAX_DEPTH_FIELD, maxDepth);
        }
        if (searchType != null) {
            builder.field(SEARCH_TYPE_FIELD, searchType);
        }
        if (entityType != null) {
            builder.field(ENTITY_TYPE_FIELD, entityType);
        }
        builder.endObject();
        return builder;
    }
}