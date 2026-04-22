/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.memorycontainer.graph;

import lombok.Builder;
import lombok.Data;

/**
 * Represents a relationship between entities in the knowledge graph
 */
@Data
@Builder
public class GraphRelationship {
    private String relationshipId;
    private String sourceEntityId;
    private String targetEntityId;
    private String relationshipType;
    private Double confidence;
}
