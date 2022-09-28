/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom_model.predict;

import org.opensearch.action.ActionType;
import org.opensearch.ml.common.transport.MLTaskResponse;

public class MLPredictModelAction extends ActionType<MLTaskResponse> {
    public static MLPredictModelAction INSTANCE = new MLPredictModelAction();
    public static final String NAME = "cluster:admin/opensearch/ml/predict_model";

    private MLPredictModelAction() {
        super(NAME, MLTaskResponse::new);
    }

}
