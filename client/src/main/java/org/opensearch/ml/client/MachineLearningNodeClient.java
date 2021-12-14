/*
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  The OpenSearch Contributors require contributions made to
 *  this file be licensed under the Apache-2.0 license or a
 *  compatible open source license.
 *
 *  Modifications Copyright OpenSearch Contributors. See
 *  GitHub history for details.
 */

package org.opensearch.ml.client;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import org.opensearch.action.ActionListener;
import org.opensearch.client.node.NodeClient;
import org.opensearch.ml.common.input.Input;
import org.opensearch.ml.common.output.Output;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskAction;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskRequest;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskResponse;
import org.opensearch.ml.common.transport.training.MLTrainingTaskAction;
import org.opensearch.ml.common.transport.training.MLTrainingTaskRequest;
import org.opensearch.ml.common.transport.training.MLTrainingTaskResponse;

@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
@RequiredArgsConstructor
public class MachineLearningNodeClient implements MachineLearningClient {

    NodeClient client;

    @Override
    public void predict(String modelId, Input input,
                        ActionListener<Output> listener) {
        validateMLInput(input, true);

        MLPredictionTaskRequest predictionRequest = MLPredictionTaskRequest.builder()
            .input(input)
            .build();

        client.execute(MLPredictionTaskAction.INSTANCE, predictionRequest, ActionListener.wrap(response -> {
            MLPredictionTaskResponse predictionResponse =
                    MLPredictionTaskResponse
                            .fromActionResponse(response);
            listener.onResponse(predictionResponse.getOutput());
        }, listener::onFailure));

    }

    @Override
    public void train(Input Input, ActionListener<Output> listener) {
        validateMLInput(Input, true);
        MLTrainingTaskRequest trainingTaskRequest = MLTrainingTaskRequest.builder()
                .input(Input)
                .build();

        client.execute(MLTrainingTaskAction.INSTANCE, trainingTaskRequest, ActionListener.wrap(response -> {
            listener.onResponse(MLTrainingTaskResponse.fromActionResponse(response).getOutput());
        }, listener::onFailure));
    }

    @Override
    public void execute(Input input, ActionListener<Output> listener) {
        MLExecuteTaskRequest executeTaskRequest = MLExecuteTaskRequest.builder()
                .input(input)
                .build();

        client.execute(MLExecuteTaskAction.INSTANCE, executeTaskRequest, ActionListener.wrap(response -> {
            listener.onResponse(MLExecuteTaskResponse.fromActionResponse(response).getOutput());
        }, listener::onFailure));
    }

    private void validateMLInput(Input Input, boolean requireInput) {
        if (Input == null) {
            throw new IllegalArgumentException("ML Input can't be null");
        }
        if(requireInput && Input.getInputDataset() == null) {
            throw new IllegalArgumentException("input data set can't be null");
        }
    }

}
