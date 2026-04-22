/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.memory;

import java.io.IOException;
import java.util.List;

import org.opensearch.core.action.ActionResponse;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.ml.common.memorycontainer.graph.GraphEntity;
import org.opensearch.ml.common.memorycontainer.graph.GraphRelationship;
import org.opensearch.ml.common.memorycontainer.graph.GraphSearchResult;

import lombok.Builder;
import lombok.Getter;

/**
 * Response for graph search operations
 */
@Getter
public class MLGraphSearchResponse extends ActionResponse implements ToXContentObject {

    public static final String ENTITIES_FIELD = "entities";
    public static final String RELATIONSHIPS_FIELD = "relationships";
    public static final String SEARCH_RESULTS_FIELD = "search_results";
    public static final String TOTAL_COUNT_FIELD = "total_count";
    public static final String QUERY_FIELD = "query";

    private List<GraphEntity> entities;
    private List<GraphRelationship> relationships;
    private List<GraphSearchResult> searchResults;
    private Integer totalCount;
    private String query;

    @Builder
    public MLGraphSearchResponse(
        List<GraphEntity> entities,
        List<GraphRelationship> relationships,
        List<GraphSearchResult> searchResults,
        Integer totalCount,
        String query
    ) {
        this.entities = entities;
        this.relationships = relationships;
        this.searchResults = searchResults;
        this.totalCount = totalCount;
        this.query = query;
    }

    public MLGraphSearchResponse(StreamInput in) throws IOException {
        super(in);

        if (in.readBoolean()) {
            this.entities = in.readList(this::readGraphEntity);
        }

        if (in.readBoolean()) {
            this.relationships = in.readList(this::readGraphRelationship);
        }

        if (in.readBoolean()) {
            this.searchResults = in.readList(this::readGraphSearchResult);
        }

        this.totalCount = in.readOptionalVInt();
        this.query = in.readOptionalString();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeBoolean(entities != null);
        if (entities != null) {
            out.writeCollection(entities, this::writeGraphEntity);
        }

        out.writeBoolean(relationships != null);
        if (relationships != null) {
            out.writeCollection(relationships, this::writeGraphRelationship);
        }

        out.writeBoolean(searchResults != null);
        if (searchResults != null) {
            out.writeCollection(searchResults, this::writeGraphSearchResult);
        }

        out.writeOptionalVInt(totalCount);
        out.writeOptionalString(query);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();

        if (entities != null && !entities.isEmpty()) {
            builder.startArray(ENTITIES_FIELD);
            for (GraphEntity entity : entities) {
                writeGraphEntityXContent(builder, entity);
            }
            builder.endArray();
        }

        if (relationships != null && !relationships.isEmpty()) {
            builder.startArray(RELATIONSHIPS_FIELD);
            for (GraphRelationship relationship : relationships) {
                writeGraphRelationshipXContent(builder, relationship);
            }
            builder.endArray();
        }

        if (searchResults != null && !searchResults.isEmpty()) {
            builder.startArray(SEARCH_RESULTS_FIELD);
            for (GraphSearchResult result : searchResults) {
                writeGraphSearchResultXContent(builder, result);
            }
            builder.endArray();
        }

        if (totalCount != null) {
            builder.field(TOTAL_COUNT_FIELD, totalCount);
        }

        if (query != null) {
            builder.field(QUERY_FIELD, query);
        }

        builder.endObject();
        return builder;
    }

    // Helper methods for streaming GraphEntity
    private GraphEntity readGraphEntity(StreamInput in) throws IOException {
        return GraphEntity.builder()
            .entityId(in.readString())
            .name(in.readString())
            .type(in.readString())
            .confidence(in.readOptionalDouble())
            .mentionCount(in.readOptionalVInt())
            .build();
    }

    private void writeGraphEntity(StreamOutput out, GraphEntity entity) throws IOException {
        out.writeString(entity.getEntityId());
        out.writeString(entity.getName());
        out.writeString(entity.getType());
        out.writeOptionalDouble(entity.getConfidence());
        out.writeOptionalVInt(entity.getMentionCount());
    }

    private void writeGraphEntityXContent(XContentBuilder builder, GraphEntity entity) throws IOException {
        builder.startObject();
        builder.field("entity_id", entity.getEntityId());
        builder.field("name", entity.getName());
        builder.field("type", entity.getType());
        if (entity.getConfidence() != null) {
            builder.field("confidence", entity.getConfidence());
        }
        if (entity.getMentionCount() != null) {
            builder.field("mention_count", entity.getMentionCount());
        }
        builder.endObject();
    }

    // Helper methods for streaming GraphRelationship
    private GraphRelationship readGraphRelationship(StreamInput in) throws IOException {
        return GraphRelationship.builder()
            .relationshipId(in.readString())
            .sourceEntityId(in.readString())
            .targetEntityId(in.readString())
            .relationshipType(in.readString())
            .confidence(in.readOptionalDouble())
            .build();
    }

    private void writeGraphRelationship(StreamOutput out, GraphRelationship relationship) throws IOException {
        out.writeString(relationship.getRelationshipId());
        out.writeString(relationship.getSourceEntityId());
        out.writeString(relationship.getTargetEntityId());
        out.writeString(relationship.getRelationshipType());
        out.writeOptionalDouble(relationship.getConfidence());
    }

    private void writeGraphRelationshipXContent(XContentBuilder builder, GraphRelationship relationship) throws IOException {
        builder.startObject();
        builder.field("relationship_id", relationship.getRelationshipId());
        builder.field("source_entity", relationship.getSourceEntityId());
        builder.field("target_entity", relationship.getTargetEntityId());
        builder.field("relationship_type", relationship.getRelationshipType());
        if (relationship.getConfidence() != null) {
            builder.field("confidence", relationship.getConfidence());
        }
        builder.endObject();
    }

    // Helper methods for streaming GraphSearchResult
    private GraphSearchResult readGraphSearchResult(StreamInput in) throws IOException {
        GraphEntity entity = readGraphEntity(in);
        return GraphSearchResult.builder()
            .entity(entity)
            .score(in.readOptionalFloat())
            .matchType(in.readOptionalString())
            .queryText(in.readOptionalString())
            .relationshipCount(in.readOptionalVInt())
            .build();
    }

    private void writeGraphSearchResult(StreamOutput out, GraphSearchResult result) throws IOException {
        writeGraphEntity(out, result.getEntity());
        out.writeOptionalFloat(result.getScore());
        out.writeOptionalString(result.getMatchType());
        out.writeOptionalString(result.getQueryText());
        out.writeOptionalVInt(result.getRelationshipCount());
    }

    private void writeGraphSearchResultXContent(XContentBuilder builder, GraphSearchResult result) throws IOException {
        builder.startObject();
        builder.field("entity");
        writeGraphEntityXContent(builder, result.getEntity());
        if (result.getScore() != null) {
            builder.field("score", result.getScore());
        }
        if (result.getMatchType() != null) {
            builder.field("match_type", result.getMatchType());
        }
        if (result.getQueryText() != null) {
            builder.field("query_text", result.getQueryText());
        }
        if (result.getRelationshipCount() != null) {
            builder.field("relationship_count", result.getRelationshipCount());
        }
        builder.endObject();
    }
}