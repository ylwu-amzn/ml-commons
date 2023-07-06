/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.remote;

import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.connector.Connector;
import org.opensearch.ml.common.dataset.TextDocsInputDataSet;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.script.ScriptService;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.ml.engine.algorithms.remote.ConnectorUtils.processInput;

public interface RemoteConnectorExecutor {

    default ModelTensorOutput executePredict(MLInput mlInput) {
        List<ModelTensors> tensorOutputs = new ArrayList<>();

        if (mlInput.getInputDataset() instanceof TextDocsInputDataSet) {
            TextDocsInputDataSet textDocsInputDataSet = (TextDocsInputDataSet) mlInput.getInputDataset();
            List textDocs = new ArrayList(textDocsInputDataSet.getDocs());
            for (int i = 0; i < textDocsInputDataSet.getDocs().size(); i++) {
                preparePayloadAndInvokeRemoteModel(MLInput.builder().algorithm(FunctionName.TEXT_EMBEDDING).inputDataset(TextDocsInputDataSet.builder().docs(textDocs).build()).build(), tensorOutputs);
                if (tensorOutputs.size() >= textDocsInputDataSet.getDocs().size()) {
                    break;
                }
                textDocs.remove(0);
            }
        } else {
            preparePayloadAndInvokeRemoteModel(mlInput, tensorOutputs);
        }
        return new ModelTensorOutput(tensorOutputs);
    }
    default void setScriptService(ScriptService scriptService){}
    ScriptService getScriptService();
    Connector getConnector();
    default void setClient(Client client){}
    default void setXContentRegistry(NamedXContentRegistry xContentRegistry){}
    default void setClusterService(ClusterService clusterService){}

    default void preparePayloadAndInvokeRemoteModel(MLInput mlInput, List<ModelTensors> tensorOutputs) {
        Connector connector = getConnector();
        RemoteInferenceInputDataSet inputData = processInput(mlInput, connector, getScriptService());

        Map<String, String> parameters = new HashMap<>();
        if (connector.getParameters() != null) {
            parameters.putAll(connector.getParameters());
        }
        if (inputData.getParameters() != null) {
            parameters.putAll(inputData.getParameters());
        }

        String payload = connector.createPredictPayload(parameters);
        connector.validatePayload(payload);
        invokeRemoteModel(mlInput, parameters, payload, tensorOutputs);
    }
    void invokeRemoteModel(MLInput mlInput, Map<String, String> parameters, String payload, List<ModelTensors> tensorOutputs);


}
