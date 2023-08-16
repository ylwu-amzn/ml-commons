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
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.input.remote.AgentMLInput;
import org.opensearch.ml.common.output.Output;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.spi.tools.ToolAnnotation;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskAction;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskRequest;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskResponse;

import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

import static java.util.concurrent.TimeUnit.SECONDS;

/**
 * This tool supports running any Agent.
 */
@Log4j2
@ToolAnnotation(AgentTool.NAME)
public class AgentTool implements Tool {
    public static final String NAME = "AgentTool";
    private final Client client;

    private String agentId;
    @Setter @Getter
    private String alias;

    @Getter @Setter
    private String description = "Use this tool to run any agent.";

    public AgentTool(Client client, String agentId) {
        this.client = client;
        this.agentId = agentId;
    }

    @Override
    public <T> T run(Map<String, String> parameters) {
        AgentMLInput agentMLInput = AgentMLInput.AgentMLInputBuilder().agentId(agentId).functionName(FunctionName.AGENT).inputDataset(RemoteInferenceInputDataSet.builder().parameters(parameters).build()).build();
        ActionRequest request = new MLExecuteTaskRequest(FunctionName.AGENT, agentMLInput, false);
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<Output> outputRef = new AtomicReference<>();
        AtomicReference<Exception> exceptionRef = new AtomicReference<>();
        client.execute(MLExecuteTaskAction.INSTANCE, request, new LatchedActionListener<MLExecuteTaskResponse>(ActionListener.wrap(r->{
            ModelTensorOutput output = (ModelTensorOutput) r.getOutput();
            outputRef.set(output);
        }, e->{
            log.error("Failed to run agent " + agentId, e);
            exceptionRef.set(e);
        }), latch));

        try {
            latch.await(50, SECONDS);
        } catch (InterruptedException e) {
            throw new IllegalStateException(e);
        }

        if (exceptionRef.get() != null) {
            throw new MLException(exceptionRef.get());
        }
        return (T)outputRef.get();
    }

    @Override
    public String getName() {
        return AgentTool.NAME;
    }

    @Override
    public boolean validate(Map<String, String> parameters) {
        return true;
    }

    public static class Factory implements Tool.Factory<AgentTool> {
        private Client client;

        private static Factory INSTANCE;
        public static Factory getInstance() {
            if (INSTANCE != null) {
                return INSTANCE;
            }
            synchronized (AgentTool.class) {
                if (INSTANCE != null) {
                    return INSTANCE;
                }
                INSTANCE = new Factory();
                return INSTANCE;
            }
        }

        public void init(Client client) {
            this.client = client;
        }

        @Override
        public AgentTool create(Map<String, Object> map) {
            return new AgentTool(client, (String)map.get("agent_id"));
        }
    }
}