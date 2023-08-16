/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.agent;

import com.google.gson.Gson;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.opensearch.ResourceNotFoundException;
import org.opensearch.action.ActionListener;
import org.opensearch.action.LatchedActionListener;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.get.GetResponse;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.bytes.BytesReference;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.agent.MLAgent;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.input.Input;
import org.opensearch.ml.common.input.remote.AgentMLInput;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.common.output.Output;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.spi.memory.Memory;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.engine.Executable;
import org.opensearch.ml.engine.annotation.Function;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

import static java.util.concurrent.TimeUnit.SECONDS;
import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.common.CommonValue.ML_AGENT_INDEX;

@Log4j2
@Data
@NoArgsConstructor
@Function(FunctionName.AGENT)
public class MLAgentExecutor implements Executable {

    private Client client;
    private Settings settings;
    private ClusterService clusterService;
    private NamedXContentRegistry xContentRegistry;
    private Map<String, Tool.Factory> toolFactories;
    private Map<String, Memory> memoryMap;

    public MLAgentExecutor(Client client, Settings settings, ClusterService clusterService, NamedXContentRegistry xContentRegistry, Map<String, Tool.Factory> toolFactories, Map<String, Memory> memoryMap) {
        this.client = client;
        this.settings = settings;
        this.clusterService = clusterService;
        this.xContentRegistry = xContentRegistry;
        this.toolFactories = toolFactories;
        this.memoryMap = memoryMap;
    }

    @Override
    public Output execute(Input input) {
        if (input == null || !(input instanceof AgentMLInput)) {
            throw new IllegalArgumentException("wrong input");
        }
        AgentMLInput agentMLInput = (AgentMLInput) input;
        String agentId = agentMLInput.getAgentId();
        RemoteInferenceInputDataSet inputDataSet = (RemoteInferenceInputDataSet)agentMLInput.getInputDataset();
        List<ModelTensors> outputs = new ArrayList<>();
        List<ModelTensor> modelTensors = new ArrayList<>();
        outputs.add(ModelTensors.builder().mlModelTensors(modelTensors).build());


        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<Exception> exceptionRef = new AtomicReference<>();
        AtomicReference<MLAgent> mlAgentRef = new AtomicReference<>();
        AtomicReference<MLOutput> outputRef = new AtomicReference<>();
        if (clusterService.state().metadata().hasIndex(ML_AGENT_INDEX)) {
            try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                GetRequest getRequest = new GetRequest(ML_AGENT_INDEX).id(agentId);
                client.get(getRequest, new LatchedActionListener(ActionListener.<GetResponse>wrap(r -> {
                    if (r.isExists()) {
                        try (XContentParser parser = createXContentParserFromRegistry(xContentRegistry, r.getSourceAsBytesRef())) {
                            ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
                            MLAgent mlAgent = MLAgent.parse(parser);
                            Object output = null;
                            if ("flow".equals(mlAgent.getType())) {
                                MLFlowAgentRunner flowAgentExecutor = new MLFlowAgentRunner(client, settings, clusterService, xContentRegistry, toolFactories);
                                output = flowAgentExecutor.run(mlAgent, inputDataSet.getParameters());
                            } else if ("cot".equals(mlAgent.getType())) {
                                MLCoTAgentRunner coTAgentExecutor = new MLCoTAgentRunner(client, settings, clusterService, xContentRegistry, toolFactories, memoryMap);
                                output = coTAgentExecutor.run(mlAgent, inputDataSet.getParameters());
                            }
                            if (output != null) {
                                Gson gson = new Gson();
                                if (output instanceof ModelTensorOutput) {
                                    ModelTensorOutput modelTensorOutput = (ModelTensorOutput) output;
                                    modelTensorOutput.getMlModelOutputs().forEach(outs -> {
                                        for (ModelTensor mlModelTensor : outs.getMlModelTensors()) {
                                            modelTensors.add(mlModelTensor);
                                        }
                                    });
                                } else if (output instanceof ModelTensor) {
                                    modelTensors.add((ModelTensor)output);
                                } else if (output instanceof List) {
                                    if (((List) output).get(0) instanceof ModelTensor) {
                                        ((List<ModelTensor>) output).forEach(mlModelTensor -> modelTensors.add(mlModelTensor));
                                    } else if (((List) output).get(0) instanceof ModelTensors) {
                                        ((List<ModelTensors>) output).forEach(outs -> {
                                            for (ModelTensor mlModelTensor : outs.getMlModelTensors()) {
                                                modelTensors.add(mlModelTensor);
                                            }
                                        });
                                    } else {
                                        Object finalOutput = output;
                                        String result = output instanceof String? (String)output : AccessController.doPrivileged((PrivilegedExceptionAction<String>) () -> gson.toJson(finalOutput));
                                        modelTensors.add(ModelTensor.builder().name("response").result(result).build());
                                    }
                                } else {
                                    Object finalOutput = output;
                                    String result = output instanceof String? (String)output : AccessController.doPrivileged((PrivilegedExceptionAction<String>) () -> gson.toJson(finalOutput));
                                    modelTensors.add(ModelTensor.builder().name("response").result(result).build());
                                }
                            }
                            outputRef.set(ModelTensorOutput.builder().mlModelOutputs(outputs).build());
                            mlAgentRef.set(mlAgent);
                        }
                    } else {
                        exceptionRef.set(new ResourceNotFoundException("Agent not found"));
                    }
                }, e -> {
                    log.error("Failed to get ML encryption master key", e);
                    exceptionRef.set(e);
                }), latch));
            }
        } else {
            exceptionRef.set(new ResourceNotFoundException("Agent not found"));
            latch.countDown();
        }

        try {
            latch.await(30, SECONDS);
        } catch (InterruptedException e) {
            throw new IllegalStateException(e);
        }
        Exception exception = exceptionRef.get();
        if (exception != null) {
            if (exception instanceof RuntimeException) {
                throw (RuntimeException) exception;
            }
            throw new MLException(exception);
        }
        return outputRef.get();
    }

    public XContentParser createXContentParserFromRegistry(NamedXContentRegistry xContentRegistry, BytesReference bytesReference)
            throws IOException {
        return XContentHelper.createParser(xContentRegistry, LoggingDeprecationHandler.INSTANCE, bytesReference, XContentType.JSON);
    }
}
