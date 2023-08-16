/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.LatchedActionListener;
import org.opensearch.client.Client;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.spi.tools.Parser;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.spi.tools.ToolAnnotation;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

import static java.util.concurrent.TimeUnit.SECONDS;

/**
 * This tool supports running any ml-commons model.
 */
@Log4j2
@ToolAnnotation(MLModelTool.NAME)
public class MLModelTool implements Tool {
    public static final String NAME = "MLModelTool";

    @Setter @Getter
    private String alias;
    @Getter @Setter
    private String description = "User this tool to run any model.";
    private Client client;
    private String modelId;
    @Setter
    private Parser inputParser;
    @Setter
    private Parser outputParser;

    public MLModelTool(Client client, String modelId) {
        this.client = client;
        this.modelId = modelId;

        outputParser = new Parser() {
            @Override
            public Object parse(Object o) {
                List<ModelTensors> mlModelOutputs = (List<ModelTensors>) o;
                return mlModelOutputs.get(0).getMlModelTensors().get(0).getDataAsMap().get("response");
            }
        };
    }


    @Override
    public <T> T run(Map<String, String> parameters) {
        RemoteInferenceInputDataSet inputDataSet = RemoteInferenceInputDataSet.builder().parameters(parameters).build();
        ActionRequest request = new MLPredictionTaskRequest(modelId, MLInput.builder().algorithm(FunctionName.REMOTE).inputDataset(inputDataSet).build());
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<Object> modelTensorsRef = new AtomicReference<>();
        client.execute(MLPredictionTaskAction.INSTANCE, request, new LatchedActionListener(ActionListener.<MLTaskResponse>wrap(r->{
            ModelTensorOutput modelTensorOutput = (ModelTensorOutput)r.getOutput();
            modelTensorOutput.getMlModelOutputs();
            if (outputParser == null) {
                modelTensorsRef.set(modelTensorOutput.getMlModelOutputs());
            } else {
                modelTensorsRef.set(outputParser.parse(modelTensorOutput.getMlModelOutputs()));
            }
        }, e->{
            log.error("Failed to run model " + modelId, e);
        }), latch));
        try {
            latch.await(30, SECONDS);
        } catch (InterruptedException e) {
            throw new IllegalStateException(e);
        }
        return (T)modelTensorsRef.get();
    }


    @Override
    public String getName() {
        return MLModelTool.NAME;
    }

    @Override
    public boolean validate(Map<String, String> parameters) {
        if (parameters == null || parameters.size() == 0) {
            return false;
        }
        return true;
    }

    public static class Factory implements Tool.Factory<MLModelTool> {
        private Client client;

        private static MLModelTool.Factory INSTANCE;
        public static MLModelTool.Factory getInstance() {
            if (INSTANCE != null) {
                return INSTANCE;
            }
            synchronized (MLModelTool.class) {
                if (INSTANCE != null) {
                    return INSTANCE;
                }
                INSTANCE = new MLModelTool.Factory();
                return INSTANCE;
            }
        }

        public void init(Client client) {
            this.client = client;
        }

        @Override
        public MLModelTool create(Map<String, Object> map) {
            return new MLModelTool(client, (String)map.get("model_id"));
        }
    }
}