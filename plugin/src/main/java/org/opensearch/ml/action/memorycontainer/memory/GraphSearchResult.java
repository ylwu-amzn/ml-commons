/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import lombok.Builder;
import lombok.Data;

/**
 * Individual search result from hybrid graph search operations
 */
@Data
@Builder
public class GraphSearchResult {
    private GraphEntity entity;
    private Float score; // Relevance score (0.0-1.0+)
    private String matchType; // "text_similarity", "relationship_expansion", etc.
    private String queryText; // Original query for context
    private Integer relationshipCount; // Number of relationships for this entity
}