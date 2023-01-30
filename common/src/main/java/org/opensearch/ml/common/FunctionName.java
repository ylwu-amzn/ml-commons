/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common;

public enum FunctionName {
    LINEAR_REGRESSION,
    KMEANS,
    AD_LIBSVM,
    SAMPLE_ALGO,
    LOCAL_SAMPLE_CALCULATOR,
    FIT_RCF,
    BATCH_RCF,
    ANOMALY_LOCALIZATION,
    RCF_SUMMARIZE,
    LOGISTIC_REGRESSION,
    TEXT_EMBEDDING,
    QUESTION_ANSWERING,
    TIME_SERIES_FORECASTING;

    public static FunctionName from(String value) {
        try {
            return FunctionName.valueOf(value);
        } catch (Exception e) {
            throw new IllegalArgumentException("Wrong function name");
        }
    }

    /**
     * Return true if model is DL model.
     * @return
     */
    public static boolean isNLPModel(FunctionName functionName) {
        if (functionName == TEXT_EMBEDDING ||
                functionName == QUESTION_ANSWERING||
                functionName == TIME_SERIES_FORECASTING) {
            return true;
        }
        return false;
    }

    public static boolean isDLPModel(FunctionName functionName) {
        if (functionName == TEXT_EMBEDDING ||
                functionName == QUESTION_ANSWERING||
                functionName == TIME_SERIES_FORECASTING) {
            return true;
        }
        return false;
    }
}
