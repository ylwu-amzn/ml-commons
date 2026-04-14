/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import lombok.Builder;
import lombok.Data;

/**
 * Represents an entity in the knowledge graph
 */
@Data
@Builder
public class GraphEntity {
    private String entityId;
    private String name;
    private String type;
    private Double confidence;
    private Integer mentionCount;
}