/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import lombok.Builder;
import lombok.Data;

/**
 * Entity extracted from conversation by LLM
 */
@Data
@Builder
public class ExtractedEntity {
    private String name;
    private String type;
    private Double confidence;
    private String normalizedName; // For ID generation

    /**
     * Normalize entity name for consistent ID generation
     */
    public String getNormalizedName() {
        if (normalizedName != null) {
            return normalizedName;
        }
        if (name == null) {
            return null;
        }
        // Normalize: lowercase, trim, remove extra spaces
        return name.toLowerCase().trim().replaceAll("\\s+", " ");
    }
}