/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.stats;

import java.util.Locale;

import org.opensearch.ml.common.FunctionName;

/**
 * Enum containing names of all stats
 */
public class StatNames {
    // cluster level stats
    public static String ML_MODEL_INDEX_STATUS = "ml_model_index_status";
    public static String ML_TASK_INDEX_STATUS = "ml_task_index_status";
    public static String ML_MODEL_COUNT = "ml_model_count";
    // node level stats
    public static String ML_NODE_EXECUTING_TASK_COUNT = "ml_node_executing_task_count";
    public static String ML_NODE_TOTAL_REQUEST_COUNT = "ml_node_total_request_count";
    public static String ML_NODE_TOTAL_FAILURE_COUNT = "ml_node_total_failure_count";
    public static String ML_NODE_TOTAL_MODEL_COUNT = "ml_node_total_model_count";
    public static String ML_NODE_TOTAL_CIRCUIT_BREAKER_TRIGGER_COUNT = "ml_node_total_circuit_breaker_trigger_count";

    // algorithm + action level stats
    public static String REQUEST_COUNT = "request_count";
    public static String FAILURE_COUNT = "failure_count";
    public static String EXECUTING_COUNT = "executing_request_count";

    public static String requestCountStat(FunctionName functionName, ActionName actionName) {
        return String.format(Locale.ROOT, "ml_%s_%s_request_count", functionName, actionName).toLowerCase(Locale.ROOT);
    }

    public static String failureCountStat(FunctionName functionName, ActionName actionName) {
        return String.format(Locale.ROOT, "ml_%s_%s_failure_count", functionName, actionName).toLowerCase(Locale.ROOT);
    }

    public static String executingRequestCountStat(FunctionName functionName, ActionName actionName) {
        return String.format(Locale.ROOT, "ml_%s_%s_executing_request_count", functionName, actionName).toLowerCase(Locale.ROOT);
    }

    public static String modelCountStat(FunctionName functionName) {
        return String.format(Locale.ROOT, "ml_%s_model_count", functionName).toLowerCase(Locale.ROOT);
    }
}
