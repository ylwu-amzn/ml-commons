/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.stats;

public enum MLClusterLevelStat {
    ML_MODEL_INDEX_STATUS,
    ML_TASK_INDEX_STATUS,
    ML_MODEL_COUNT;

    public static MLClusterLevelStat from(String value) {
        try {
            return MLClusterLevelStat.valueOf(value);
        } catch (Exception e) {
            throw new IllegalArgumentException("Wrong ML cluster level stat");
        }
    }
}
