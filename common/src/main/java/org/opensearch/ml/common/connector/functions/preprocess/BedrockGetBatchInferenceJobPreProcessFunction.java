/*
 *
 *  * Copyright OpenSearch Contributors
 *  * SPDX-License-Identifier: Apache-2.0
 *
 */

package org.opensearch.ml.common.connector.functions.preprocess;

import org.opensearch.ml.common.dataset.TextDocsInputDataSet;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.ml.common.utils.StringUtils.convertScriptStringToJsonString;

/**
 * This class provides a pre-processing function for bedrock batch inference job input.
 * It takes an instance of {@link MLInput} as input and returns an instance of {@link RemoteInferenceInputDataSet}.
 * The input data is expected to be of type {@link RemoteInferenceInputDataSet}, which must have jobArn parameter.
 * The function validates the input data and then processes it to create a {@link RemoteInferenceInputDataSet} object.
 */
public class BedrockGetBatchInferenceJobPreProcessFunction extends ConnectorPreProcessFunction {

    public static final String JOB_ARN = "jobArn";
    public static final String PROCESSED_JOB_ARN = "processedJobArn";

    public BedrockGetBatchInferenceJobPreProcessFunction() {
        this.returnDirectlyForRemoteInferenceInput = false;
    }

    @Override
    public void validate(MLInput mlInput) {
        if (!(mlInput.getInputDataset() instanceof RemoteInferenceInputDataSet)) {
            throw new IllegalArgumentException("Wrong input dataset type");
        }
        RemoteInferenceInputDataSet inputData = (RemoteInferenceInputDataSet) mlInput.getInputDataset();
        if (inputData == null) {
            throw new IllegalArgumentException("No input dataset provided");
        }
        String jobArn = inputData.getParameters().get(JOB_ARN);
        if (jobArn == null) {
            throw new IllegalArgumentException("No jobArn provided");
        }
    }

    /**
     *  @param mlInput The input data to be processed.
     *  This method is to escape slash in jobArn.
     */
    @Override
    public RemoteInferenceInputDataSet process(MLInput mlInput) {
        RemoteInferenceInputDataSet inputData = (RemoteInferenceInputDataSet) mlInput.getInputDataset();
        String jobArn = inputData.getParameters().get(JOB_ARN);
        inputData.getParameters().put(PROCESSED_JOB_ARN, jobArn.replace("/", "%2F"));
        return inputData;
    }
}
