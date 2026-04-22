/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import java.util.List;

import org.opensearch.ml.common.memorycontainer.graph.GraphEntity;
import org.opensearch.ml.common.memorycontainer.graph.GraphRelationship;

import lombok.Builder;
import lombok.Data;

/**
 * Result of graph traversal operations containing entities and relationships
 */
@Data
@Builder
public class GraphTraversalResult {
    private List<GraphEntity> entities;
    private List<GraphRelationship> relationships;
    private String centerEntityId; // The entity that was the starting point
    private Integer traversalDepth; // Maximum depth traversed
}