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


import org.opensearch.action.ActionFuture;
import org.opensearch.action.ActionListener;
import org.opensearch.action.support.PlainActionFuture;
import org.opensearch.ml.common.input.Input;
import org.opensearch.ml.common.output.Output;

/**
 * A client to provide interfaces for machine learning jobs. This will be used by other plugins.
 */
public interface MachineLearningClient {

    /**
     * Do prediction machine learning job
     * @param modelId the trained model id
     * @param Input ML input
     * @return
     */
    default ActionFuture<Output> predict(String modelId, Input Input) {
        PlainActionFuture<Output> actionFuture = PlainActionFuture.newFuture();
        predict(modelId, Input, actionFuture);
        return actionFuture;
    }

    /**
     * Do prediction machine learning job
     * @param modelId the trained model id
     * @param Input ML input
     * @param listener a listener to be notified of the result
     */
    void predict(String modelId, Input Input, ActionListener<Output> listener);

    /**
     *  Do the training machine learning job. The training job will be always async process. The job id will be returned in this method.
     * @param Input ML input
     * @return the result future
     */
    default ActionFuture<Output> train(Input Input) {
        PlainActionFuture<Output> actionFuture = PlainActionFuture.newFuture();
        train(Input, actionFuture);
        return actionFuture;
    }


    /**
     * Do the training machine learning job. The training job will be always async process. The job id will be returned in this method.
     * @param Input ML input
     * @param listener a listener to be notified of the result
     */
    void train(Input Input, ActionListener<Output> listener);

    /**
     * Execute function and return ActionFuture.
     * @param input input data
     * @return ActionFuture of output
     */
    default ActionFuture<Output> execute(Input input) {
        PlainActionFuture<Output> actionFuture = PlainActionFuture.newFuture();
        execute(input, actionFuture);
        return actionFuture;
    }

    /**
     * Execute function and return output in listener
     * @param input input data
     * @param listener action listener
     */
    void execute(Input input, ActionListener<Output> listener);
}
