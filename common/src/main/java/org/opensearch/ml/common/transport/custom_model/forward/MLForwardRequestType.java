/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom_model.forward;

public enum MLForwardRequestType {
    LOAD_MODEL_DONE,
    UPLOAD_MODEL,
    PREDICT_MODEL;
}
