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

import lombok.extern.log4j.Log4j2;
import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskResponse;
import org.opensearch.ml.common.transport.training.MLTrainingTaskRequest;
import org.opensearch.ml.task.MLTrainAndPredictTaskRunner;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

@Log4j2
public class MLTrainAndPredictionTaskExecutionTransportAction extends
    HandledTransportAction<ActionRequest, MLPredictionTaskResponse> {
    private final MLTrainAndPredictTaskRunner mlTrainAndPredictTaskRunner;
    private final TransportService transportService;

    @Inject
    public MLTrainAndPredictionTaskExecutionTransportAction(
        ActionFilters actionFilters,
        TransportService transportService,
        MLTrainAndPredictTaskRunner mlTrainAndPredictTaskRunner
    ) {
        super(MLTrainAndPredictionTaskExecutionAction.NAME, transportService, actionFilters, MLTrainingTaskRequest::new);
        this.mlTrainAndPredictTaskRunner = mlTrainAndPredictTaskRunner;
        this.transportService = transportService;
    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<MLPredictionTaskResponse> listener) {
        log.info("Received train and predict request " + request);
        MLTrainingTaskRequest mlTrainingTaskRequest = MLTrainingTaskRequest.fromActionRequest(request);
        mlTrainAndPredictTaskRunner.startTrainAndPredictionTask(mlTrainingTaskRequest, listener);
    }
}
