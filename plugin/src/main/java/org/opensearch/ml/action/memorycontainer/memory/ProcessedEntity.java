/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import lombok.Builder;
import lombok.Data;

/**
 * Entity after processing and deduplication
 */
@Data
@Builder
public class ProcessedEntity {
    private String entityId; // Generated ID using entity_scope
    private String name;
    private String type;
    private Double confidence;
    private boolean isNewEntity; // vs merged with existing
    private String existingEntityId; // If merged
    private int mentionCount; // Number of times mentioned
}