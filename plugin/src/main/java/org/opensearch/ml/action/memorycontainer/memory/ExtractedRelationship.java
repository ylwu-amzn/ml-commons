/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import lombok.Builder;
import lombok.Data;

/**
 * Relationship extracted from conversation by LLM
 */
@Data
@Builder
public class ExtractedRelationship {
    private String sourceEntity;
    private String targetEntity;
    private String type;
    private Double confidence;

    /**
     * Generate a unique relationship ID for storage
     */
    public String generateRelationshipId() {
        if (sourceEntity == null || targetEntity == null || type == null) {
            return null;
        }
        // Create consistent ID regardless of direction for symmetric relationships
        String normalizedSource = sourceEntity.toLowerCase().trim();
        String normalizedTarget = targetEntity.toLowerCase().trim();
        String normalizedType = type.toLowerCase().trim();

        return String.format("rel:%s:%s:%s", normalizedSource, normalizedTarget, normalizedType);
    }
}