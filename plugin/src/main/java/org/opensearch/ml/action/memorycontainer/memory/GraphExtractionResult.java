/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import java.util.List;

import lombok.Builder;
import lombok.Data;

/**
 * Result of entity and relationship extraction from conversations
 */
@Data
@Builder
public class GraphExtractionResult {
    private List<ExtractedEntity> entities;
    private List<ExtractedRelationship> relationships;
}