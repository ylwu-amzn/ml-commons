/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 *
 */

package org.opensearch.ml.action.trainpredict;

import org.opensearch.action.ActionType;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskResponse;

public class MLTrainAndPredictionTaskExecutionAction extends ActionType<MLPredictionTaskResponse> {
    public static MLTrainAndPredictionTaskExecutionAction INSTANCE = new MLTrainAndPredictionTaskExecutionAction();
    public static final String NAME = "cluster:admin/opensearch/ml/trainAndPrediction/execution";

    private MLTrainAndPredictionTaskExecutionAction() {
        super(NAME, MLPredictionTaskResponse::new);
    }
}
