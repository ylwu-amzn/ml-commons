/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.stats;

public enum MLActionLevelStat {
    ML_ACTION_REQUEST_COUNT,
    ML_ACTION_FAILURE_COUNT;

    public static MLActionLevelStat from(String value) {
        try {
            return MLActionLevelStat.valueOf(value);
        } catch (Exception e) {
            throw new IllegalArgumentException("Wrong ML action level stat");
        }
    }
}
