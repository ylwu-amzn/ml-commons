/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.agent;

import static org.apache.commons.text.StringEscapeUtils.escapeJson;
import static org.opensearch.ml.common.conversation.ActionConstants.ADDITIONAL_INFO_FIELD;
import static org.opensearch.ml.common.conversation.ActionConstants.AI_RESPONSE_FIELD;
import static org.opensearch.ml.common.conversation.ActionConstants.MEMORY_ID;
import static org.opensearch.ml.common.conversation.ActionConstants.PARENT_INTERACTION_ID_FIELD;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.getMessageHistoryLimit;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.getToolName;
import static org.opensearch.ml.engine.algorithms.agent.MLAgentExecutor.QUESTION;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.text.StringSubstitutor;
import org.opensearch.action.StepListener;
import org.opensearch.action.update.UpdateResponse;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.Strings;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.agent.MLAgent;
import org.opensearch.ml.common.agent.MLMemorySpec;
import org.opensearch.ml.common.agent.MLToolSpec;
import org.opensearch.ml.common.conversation.ActionConstants;
import org.opensearch.ml.common.conversation.Interaction;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.spi.memory.Memory;
import org.opensearch.ml.common.spi.memory.Message;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.utils.StringUtils;
import org.opensearch.ml.engine.memory.ConversationIndexMemory;
import org.opensearch.ml.engine.memory.ConversationIndexMessage;
import org.opensearch.ml.repackage.com.google.common.annotations.VisibleForTesting;
import org.opensearch.ml.repackage.com.google.common.collect.ImmutableMap;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;

@Log4j2
@Data
@NoArgsConstructor
public class MLFlowAgentRunner implements MLAgentRunner {

    public static final String CHAT_HISTORY = "chat_history";
    private Client client;
    private Settings settings;
    private ClusterService clusterService;
    private NamedXContentRegistry xContentRegistry;
    private Map<String, Tool.Factory> toolFactories;
    private Map<String, Memory.Factory> memoryFactoryMap;

    public MLFlowAgentRunner(
        Client client,
        Settings settings,
        ClusterService clusterService,
        NamedXContentRegistry xContentRegistry,
        Map<String, Tool.Factory> toolFactories,
        Map<String, Memory.Factory> memoryFactoryMap
    ) {
        this.client = client;
        this.settings = settings;
        this.clusterService = clusterService;
        this.xContentRegistry = xContentRegistry;
        this.toolFactories = toolFactories;
        this.memoryFactoryMap = memoryFactoryMap;
    }

    @Override
    public void run(MLAgent mlAgent, Map<String, String> params, ActionListener<Object> listener) {
        String appType = mlAgent.getAppType();
        String memoryId = params.get(MLAgentExecutor.MEMORY_ID);
        String parentInteractionId = params.get(MLAgentExecutor.PARENT_INTERACTION_ID);
        if (appType == null || mlAgent.getMemory() == null) {
            runAgent(mlAgent, params, listener, null, memoryId, parentInteractionId);
            return;
        }

        //TODO: refactor to extract common part with chat agent
        String memoryType = mlAgent.getMemory().getType();
        String title = params.get(QUESTION);
        int messageHistoryLimit = getMessageHistoryLimit(params);

        ConversationIndexMemory.Factory conversationIndexMemoryFactory = (ConversationIndexMemory.Factory) memoryFactoryMap.get(memoryType);
        conversationIndexMemoryFactory.create(title, memoryId, appType, ActionListener.wrap(memory -> {
            memory.getMessages(ActionListener.<List<Interaction>>wrap(r -> {
                List<Message> messageList = new ArrayList<>();
                for (Interaction next : r) {
                    String question = next.getInput();
                    String response = next.getResponse();
                    // As we store the conversation with empty response first and then update when have final answer,
                    // filter out those in-flight requests when run in parallel
                    if (Strings.isNullOrEmpty(response)) {
                        continue;
                    }
                    messageList
                        .add(
                            ConversationIndexMessage
                                .conversationIndexMessageBuilder()
                                .sessionId(memory.getConversationId())
                                .question(question)
                                .response(response)
                                .build()
                        );
                }

                StringBuilder chatHistoryBuilder = new StringBuilder();
                if (messageList.size() > 0) {
                    chatHistoryBuilder.append("Below is Chat History between Human and AI which sorted by time with asc order:\n");
                    for (Message message : messageList) {
                        chatHistoryBuilder.append(message.toString()).append("\n");
                    }
                    params.put(CHAT_HISTORY, chatHistoryBuilder.toString());
                }

                runAgent(mlAgent, params, listener, memory, memory.getConversationId(), parentInteractionId);
            }, e -> {
                log.error("Failed to get chat history", e);
                listener.onFailure(e);
            }), messageHistoryLimit);
        }, listener::onFailure));
    }

    private void runAgent(
        MLAgent mlAgent,
        Map<String, String> params,
        ActionListener<Object> listener,
        ConversationIndexMemory memory,
        String memoryId,
        String parentInteractionId
    ) {
        List<MLToolSpec> toolSpecs = mlAgent.getTools();
        StepListener<Object> firstStepListener = null;
        Tool firstTool = null;
        List<ModelTensor> flowAgentOutput = new ArrayList<>();
        Map<String, String> firstToolExecuteParams = null;
        StepListener<Object> previousStepListener = null;
        Map<String, Object> additionalInfo = new ConcurrentHashMap<>();
        if (toolSpecs == null || toolSpecs.size() == 0) {
            listener.onFailure(new IllegalArgumentException("no tool configured"));
            return;
        }
        AtomicInteger traceNumber = new AtomicInteger(0);
        if (memory != null) {
            flowAgentOutput.add(ModelTensor.builder().name(MEMORY_ID).result(memoryId).build());
            flowAgentOutput.add(ModelTensor.builder().name(PARENT_INTERACTION_ID_FIELD).result(parentInteractionId).build());
        }

        MLMemorySpec memorySpec = mlAgent.getMemory();
        for (int i = 0; i <= toolSpecs.size(); i++) {
            if (i == 0) {
                MLToolSpec toolSpec = toolSpecs.get(i);
                Tool tool = createTool(toolSpec);
                firstStepListener = new StepListener<>();
                previousStepListener = firstStepListener;
                firstTool = tool;
                firstToolExecuteParams = getToolExecuteParams(toolSpec, params);
            } else {
                MLToolSpec previousToolSpec = toolSpecs.get(i - 1);
                StepListener<Object> nextStepListener = new StepListener<>();
                int finalI = i;
                previousStepListener.whenComplete(output -> {
                    processOutput(params, listener, memory, memoryId, parentInteractionId, toolSpecs, flowAgentOutput, additionalInfo, traceNumber, memorySpec, previousToolSpec, finalI, output);
                    if (finalI == toolSpecs.size()) {
                        return;
                    }

                    MLToolSpec toolSpec = toolSpecs.get(finalI);
                    Tool tool = createTool(toolSpec);
                    if (finalI < toolSpecs.size()) {
                        tool.run(getToolExecuteParams(toolSpec, params), nextStepListener);
                    }

                }, e -> {
                    log.error("Failed to run flow agent", e);
                    listener.onFailure(e);
                });
                previousStepListener = nextStepListener;
            }
        }
        if (toolSpecs.size() == 1) {
            firstTool.run(firstToolExecuteParams, ActionListener.wrap(output -> {
                MLToolSpec toolSpec = toolSpecs.get(0);
                processOutput(params, listener, memory, memoryId, parentInteractionId,
                        toolSpecs, flowAgentOutput, additionalInfo, traceNumber, memorySpec, toolSpec, 1, output);

//                String key = toolSpec.getName();
//                String outputKey = toolSpec.getName() != null ? toolSpec.getName() + ".output" : toolSpec.getType() + ".output";
//
//                String outputResponse = parseResponse(output);
//                params.put(outputKey, escapeJson(outputResponse));
//
//                // if (toolSpec.isIncludeOutputInAgentResponse()) {
//                if (output instanceof ModelTensorOutput) {
//                    flowAgentOutput.addAll(((ModelTensorOutput) output).getMlModelOutputs().get(0).getMlModelTensors());
//                } else {
//                    String result = output instanceof String
//                        ? (String) output
//                        : AccessController.doPrivileged((PrivilegedExceptionAction<String>) () -> StringUtils.toJson(output));
//
//                    ModelTensor stepOutput = ModelTensor.builder().name(key).result(result).build();
//                    flowAgentOutput.add(stepOutput);
//                }
//
//                if (toolSpec.isIncludeOutputInAgentResponse()) {
//                    additionalInfo.put(outputKey, outputResponse);
//                }
//                // }
//
//                if (memory == null) {
//                    if (memoryId == null || parentInteractionId == null || memorySpec == null || memorySpec.getType() == null) {
//                        listener.onResponse(flowAgentOutput);
//                    } else {
//                        ActionListener updateListener = ActionListener.<UpdateResponse>wrap(r -> {
//                            log.info("Updated additional info for interaction " + r.getId() + " of flow agent.");
//                            listener.onResponse(flowAgentOutput);
//                        }, e -> {
//                            log.error("Failed to update root interaction", e);
//                            listener.onResponse(flowAgentOutput);
//                        });
//                        updateMemoryWithListener(additionalInfo, memorySpec, memoryId, parentInteractionId, updateListener);
//                    }
//                } else {
//                    ConversationIndexMessage finalMessage = ConversationIndexMessage
//                        .conversationIndexMessageBuilder()
//                        .type(memory.getType())
//                        .question(params.get(QUESTION))
//                        .response(outputResponse)
//                        .finalAnswer(true)
//                        .sessionId(memoryId)
//                        .build();
//                    memory.save(finalMessage, parentInteractionId, traceNumber.addAndGet(1), null, ActionListener.wrap(r -> {
//                        log.info("saved last trace for interaction " + parentInteractionId + " of flow agent");
//
//                        Map<String, Object> updateContent = Map.of(AI_RESPONSE_FIELD, flowAgentOutput, ADDITIONAL_INFO_FIELD, additionalInfo);
//                        memory.update(parentInteractionId, updateContent, updateListener);
//
//                    }, e -> {
//                        log.error("11111 Failed to update root interaction ", e);
//                        // listener.onResponse(flowAgentOutput);
//                        listener.onFailure(e);
//                    }));
//                }
            }, e -> { listener.onFailure(e); }));
        } else {
            firstTool.run(firstToolExecuteParams, firstStepListener);
        }
    }

    private void processOutput(Map<String, String> params,
                                  ActionListener<Object> listener,
                                  ConversationIndexMemory memory,
                                  String memoryId,
                                  String parentInteractionId,
                                  List<MLToolSpec> toolSpecs,
                                  List<ModelTensor> flowAgentOutput,
                                  Map<String, Object> additionalInfo,
                                  AtomicInteger traceNumber,
                                  MLMemorySpec memorySpec,
                                  MLToolSpec previousToolSpec,
                                  int finalI,
                                  Object output) throws IOException, PrivilegedActionException {
        String toolName = getToolName(previousToolSpec);
        String outputKey = toolName + ".output";
        String outputResponse = parseResponse(output);
        params.put(outputKey, escapeJson(outputResponse));

        if (previousToolSpec.isIncludeOutputInAgentResponse() || finalI == toolSpecs.size()) {
            if (output instanceof ModelTensorOutput) {
                flowAgentOutput.addAll(((ModelTensorOutput) output).getMlModelOutputs().get(0).getMlModelTensors());
            } else {
                String result = output instanceof String
                    ? (String) output
                    : AccessController.doPrivileged((PrivilegedExceptionAction<String>) () -> StringUtils.toJson(output));

                ModelTensor stepOutput = ModelTensor.builder().name(toolName).result(result).build();
                flowAgentOutput.add(stepOutput);
            }
            if (memory == null) {
                additionalInfo.put(outputKey, outputResponse);
            }
        }

        if (finalI == toolSpecs.size()) {
            ActionListener updateListener = ActionListener.<UpdateResponse>wrap(r -> {
                log.info("Updated additional info for interaction " + r.getId() + " of flow agent.");
                listener.onResponse(flowAgentOutput);
            }, e -> {
                log.error("Failed to update root interaction", e);
                listener.onResponse(flowAgentOutput);
            });
            if (memory == null) {
                if (memoryId == null || parentInteractionId == null || memorySpec == null || memorySpec.getType() == null) {
                    listener.onResponse(flowAgentOutput);
                } else {
                    updateMemoryWithListener(additionalInfo, memorySpec, memoryId, parentInteractionId, updateListener);
                }
            } else {
                ConversationIndexMessage finalMessage = ConversationIndexMessage
                    .conversationIndexMessageBuilder()
                    .type(memory.getType())
                    .question(params.get(QUESTION))
                    .response(outputResponse)
                    .finalAnswer(true)
                    .sessionId(memoryId)
                    .build();
                memory.save(finalMessage, parentInteractionId, traceNumber.addAndGet(1), null, ActionListener.wrap(r -> {
                    log.info("saved last trace for interaction " + parentInteractionId + " of flow agent");
                    Map<String, Object> updateContent = Map.of(AI_RESPONSE_FIELD, flowAgentOutput, ADDITIONAL_INFO_FIELD, additionalInfo);
                    memory.update(parentInteractionId, updateContent, updateListener);
                }, e -> {
                    log.error("Failed to update root interaction ", e);
                    listener.onFailure(e);
                }));

            }
        }
    }

    @VisibleForTesting
    void updateMemoryWithListener(
        Map<String, Object> additionalInfo,
        MLMemorySpec memorySpec,
        String memoryId,
        String interactionId,
        ActionListener listener
    ) {
        if (memoryId == null || interactionId == null || memorySpec == null || memorySpec.getType() == null) {
            return;
        }
        ConversationIndexMemory.Factory conversationIndexMemoryFactory = (ConversationIndexMemory.Factory) memoryFactoryMap
            .get(memorySpec.getType());
        conversationIndexMemoryFactory
            .create(
                memoryId,
                ActionListener
                    .wrap(
                        memory -> updateInteractionWithListener(additionalInfo, interactionId, memory, listener),
                        e -> log.error("Failed create memory from id: " + memoryId, e)
                    )
            );
    }

    @VisibleForTesting
    void updateInteraction(Map<String, Object> additionalInfo, String interactionId, ConversationIndexMemory memory) {
        memory
            .getMemoryManager()
            .updateInteraction(
                interactionId,
                ImmutableMap.of(ActionConstants.ADDITIONAL_INFO_FIELD, additionalInfo),
                ActionListener.<UpdateResponse>wrap(updateResponse -> {
                    log.info("Updated additional info for interaction ID: " + interactionId);
                }, e -> { log.error("Failed to update root interaction", e); })
            );
    }

    @VisibleForTesting
    void updateInteractionWithListener(
        Map<String, Object> additionalInfo,
        String interactionId,
        ConversationIndexMemory memory,
        ActionListener listener
    ) {
        memory
            .getMemoryManager()
            .updateInteraction(interactionId, ImmutableMap.of(ActionConstants.ADDITIONAL_INFO_FIELD, additionalInfo), listener);
    }

    @VisibleForTesting
    String parseResponse(Object output) throws IOException {
        if (output instanceof List && !((List) output).isEmpty() && ((List) output).get(0) instanceof ModelTensors) {
            ModelTensors tensors = (ModelTensors) ((List) output).get(0);
            return tensors.toXContent(JsonXContent.contentBuilder(), null).toString();
        } else if (output instanceof ModelTensor) {
            return ((ModelTensor) output).toXContent(JsonXContent.contentBuilder(), null).toString();
        } else if (output instanceof ModelTensorOutput) {
            return ((ModelTensorOutput) output).toXContent(JsonXContent.contentBuilder(), null).toString();
        } else {
            if (output instanceof String) {
                return (String) output;
            } else {
                return StringUtils.toJson(output);
            }
        }
    }

    @VisibleForTesting
    Tool createTool(MLToolSpec toolSpec) {
        Map<String, String> toolParams = new HashMap<>();
        if (toolSpec.getParameters() != null) {
            toolParams.putAll(toolSpec.getParameters());
        }
        if (!toolFactories.containsKey(toolSpec.getType())) {
            throw new IllegalArgumentException("Tool not found: " + toolSpec.getType());
        }
        Tool tool = toolFactories.get(toolSpec.getType()).create(toolParams);
        if (toolSpec.getName() != null) {
            tool.setName(toolSpec.getName());
        }

        if (toolSpec.getDescription() != null) {
            tool.setDescription(toolSpec.getDescription());
        }
        return tool;
    }

    @VisibleForTesting
    Map<String, String> getToolExecuteParams(MLToolSpec toolSpec, Map<String, String> params) {
        Map<String, String> executeParams = new HashMap<>();
        if (toolSpec.getParameters() != null) {
            executeParams.putAll(toolSpec.getParameters());
        }
        for (String key : params.keySet()) {
            String toBeReplaced = null;
            if (key.startsWith(toolSpec.getType() + ".")) {
                toBeReplaced = toolSpec.getType() + ".";
            }
            if (toolSpec.getName() != null && key.startsWith(toolSpec.getName() + ".")) {
                toBeReplaced = toolSpec.getName() + ".";
            }
            if (toBeReplaced != null) {
                executeParams.put(key.replace(toBeReplaced, ""), params.get(key));
            } else {
                executeParams.put(key, params.get(key));
            }
        }

        if (executeParams.containsKey("input")) {
            String input = executeParams.get("input");
            StringSubstitutor substitutor = new StringSubstitutor(executeParams, "${parameters.", "}");
            input = substitutor.replace(input);
            executeParams.put("input", input);
        }
        return executeParams;
    }
}
