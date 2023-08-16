/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.agent;

import com.google.common.collect.ImmutableMap;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.text.StringSubstitutor;
import org.opensearch.action.ActionRequest;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.agent.LLMSpec;
import org.opensearch.ml.common.agent.MLAgent;
import org.opensearch.ml.common.agent.MLToolSpec;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.input.remote.RemoteInferenceMLInput;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.spi.memory.Memory;
import org.opensearch.ml.common.spi.memory.Message;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.engine.memory.ConversationBufferWindowMemory;
import org.opensearch.ml.engine.memory.ConversationMessage;
import org.opensearch.ml.engine.tools.MLModelTool;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.opensearch.ml.engine.utils.ScriptUtils.gson;

@Log4j2
@Data
@NoArgsConstructor
public class MLCoTAgentRunner {

    public static final String SESSION_ID = "session_id";
    public static final String PROMPT_PREFIX = "prompt_prefix";
    public static final String LLM_TOOL_PROMPT_PREFIX = "LanguageModelTool.prompt_prefix";
    public static final String LLM_TOOL_PROMPT_SUFFIX = "LanguageModelTool.prompt_suffix";
    public static final String PROMPT_SUFFIX = "prompt_suffix";
    public static final String TOOLS = "tools";
    public static final String TOOL_DESCRIPTIONS = "tool_descriptions";
    public static final String TOOL_NAMES = "tool_names";
    public static final String OS_INDICES = "opensearch_indices";
    public static final String EXAMPLES = "examples";
    public static final String QUESTION = "question";
    public static final String SCRATCHPAD = "scratchpad";
    public static final String CHAT_HISTORY = "chat_history";
    public static final String CONTEXT = "context";
    public static final String PROMPT = "prompt";

    private Client client;
    private Settings settings;
    private ClusterService clusterService;
    private NamedXContentRegistry xContentRegistry;
    private Map<String, Tool.Factory> toolFactories;
    private Map<String, Memory> memoryMap;

    public MLCoTAgentRunner(Client client, Settings settings, ClusterService clusterService, NamedXContentRegistry xContentRegistry, Map<String, Tool.Factory> toolFactories, Map<String, Memory> memoryMap) {
        this.client = client;
        this.settings = settings;
        this.clusterService = clusterService;
        this.xContentRegistry = xContentRegistry;
        this.toolFactories = toolFactories;
        this.memoryMap = memoryMap;
    }

    public Object run(MLAgent mlAgent, Map<String, String> params) {
        List<MLToolSpec> toolSpecs = mlAgent.getTools();
        ConversationBufferWindowMemory memory = null;
        String sessionId = params.get(SESSION_ID);
        if (mlAgent.getMemory() != null) {
            String memoryType = mlAgent.getMemory().getType();
            if (!memoryType.startsWith("conversation") || !this.memoryMap.containsKey(memoryType)) {
                throw new IllegalArgumentException("Invalid memory type");
            }
            memory = (ConversationBufferWindowMemory)memoryMap.get(memoryType);
            if (sessionId == null) {
                sessionId = UUID.randomUUID().toString();
            }
            String onlyIncludeFinalAnswerInChatHistory = params.get("only_include_final_answer_in_chat_history");
            String onlyIncludeObservationInChatHistory = params.get("only_include_observation_in_chat_history");
            Message[] messages = memory.getMessages(sessionId);
            if ("true".equals(onlyIncludeFinalAnswerInChatHistory) && messages != null && messages.length > 0) {
                List<Message> messageList = new ArrayList<>();
                Message question = null;
                for (Message message : messages) {
                    if ("Human".equals(message.getType())) {
                        question = message;
                    }
                    if (message instanceof ConversationMessage && ((ConversationMessage)message).isFinalAnswer()) {
                        if (question != null) {
                            messageList.add(question);
                        }
                        messageList.add(message);
                        question = null;
                    }
                }
                messages = messageList.toArray(new Message[0]);
            }
            if ("true".equals(onlyIncludeObservationInChatHistory) && messages != null && messages.length > 0) {
                List<Message> messageList = new ArrayList<>();
                for (Message message : messages) {
                    if ("Tool".equals(message.getType())) {
                        messageList.add(message);
                    }
                }
                messages = messageList.toArray(new Message[0]);
            }
            StringBuilder chatHistoryBuilder = new StringBuilder();
            if (messages != null && messages.length > 0) {
                chatHistoryBuilder.append("Below is Chat History between Human and AI in <chat_history>:\n");
                chatHistoryBuilder.append("<chat_history>\n");
                for (Message message : messages) {
                    chatHistoryBuilder.append("<message>\n");
                    chatHistoryBuilder.append(message.toString()).append("\n");
                    chatHistoryBuilder.append("</message>\n");
                }
                chatHistoryBuilder.append("</chat_history>\n");
                params.put(CHAT_HISTORY, chatHistoryBuilder.toString());
            }

        }

        LLMSpec llm = mlAgent.getLlm();
        Map<String, Tool> tools = new HashMap<>();
        for (int i = 0 ;i<toolSpecs.size(); i++) {
            MLToolSpec toolSpec = toolSpecs.get(i);
            Map<String, String> toolParams = new HashMap<>();
            Map<String, String> executeParams = new HashMap<>();
            if (toolSpec.getParameters() != null) {
                toolParams.putAll(toolSpec.getParameters());
                executeParams.putAll(toolSpec.getParameters());
            }
            for (String key : params.keySet()) {
                if (key.startsWith(toolSpec.getName() + ".")) {
                    executeParams.put(key.replace(toolSpec.getName()+".", ""), params.get(key));
                }
            }
            Tool tool = toolFactories.get(toolSpec.getName()).create(toolParams);
            tool.setAlias(toolSpec.getAlias());

            if (toolSpec.getDescription() != null) {
                tool.setDescription(toolSpec.getDescription());
            }
            String toolName = Optional.ofNullable(toolSpec.getAlias()).orElse(toolSpec.getName());
            tools.put(toolName, tool);
        }

        return runCoT(llm, tools, params, memory, sessionId);
    }

    private ModelTensorOutput runCoT(LLMSpec llm, Map<String, Tool> tools, Map<String, String> parameters, ConversationBufferWindowMemory memory, String sessionId) {
        String question = parameters.get(QUESTION);
        Map<String, String> tmpParameters = new HashMap<>();
        if (llm.getParameters() != null) {
            tmpParameters.putAll(llm.getParameters());
        }
        tmpParameters.putAll(parameters);
        if (!tmpParameters.containsKey("stop")) {
            tmpParameters.put("stop", gson.toJson(new String[]{"\nObservation:", "\n\tObservation:"}));
        }
        if (!tmpParameters.containsKey("stop_sequences")) {
            tmpParameters.put("stop_sequences", gson.toJson(new String[]{"\n\nHuman:", "\nObservation:", "\n\tObservation:","\nObservation", "\n\tObservation", "\n\nQuestion"}));
        }

        String prompt = parameters.get(PROMPT);
        if (prompt == null) {
            prompt = PromptTemplate.AGENT_TEMPLATE_WITH_CONTEXT;
        }

        List<String> inputTools = new ArrayList<>();
        if (parameters.containsKey(TOOLS)) {
            inputTools = gson.fromJson(parameters.get(TOOLS), List.class);
        } else {
            for (Map.Entry<String, Tool> entry : tools.entrySet()) {
                String toolName = Optional.ofNullable(entry.getValue().getAlias()).orElse(entry.getValue().getName());
                inputTools.add(toolName);
            }
        }

        prompt = addPrefixSuffixToPrompt(parameters, prompt);
        prompt = addToolsToPrompt(tools, parameters, inputTools, prompt);
        prompt = addIndicesToPrompt(parameters, prompt);
        prompt = addExamplesToPrompt(parameters, prompt);
        prompt = addChatHistoryToPrompt(parameters, prompt);
        prompt = addContextToPrompt(parameters, prompt);

        tmpParameters.put(PROMPT, prompt);

        List<ModelTensors> modelTensors = new ArrayList<>();
        ModelTensorOutput modelTensorOutput = ModelTensorOutput.builder().mlModelOutputs(modelTensors).build();


        List<ModelTensors> cotModelTensors = new ArrayList<>();
        cotModelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().name(SESSION_ID)
                .result(sessionId).build())).build());

        StringBuilder scratchpadBuilder = new StringBuilder();
        StringSubstitutor tmpSubstitutor = new StringSubstitutor(ImmutableMap.of(SCRATCHPAD, scratchpadBuilder.toString()), "${parameters.", "}");
        String newPrompt = tmpSubstitutor.replace(prompt);
        tmpParameters.put(PROMPT, newPrompt);

        if (memory != null) {
            memory.save(sessionId, ConversationMessage.conversationMessageBuilder().type("Human").content(question).build());
        }
        String maxIteration = Optional.ofNullable(llm.getParameters().get("max_iteration")).orElse("3");
        for (int i = 0 ;i < Integer.parseInt(maxIteration); i++) {
            StringBuilder sessionMsgAnswerBuilder = new StringBuilder("");

            ActionRequest request = new MLPredictionTaskRequest(llm.getModelId(), RemoteInferenceMLInput.builder()
                    .algorithm(FunctionName.REMOTE)
                    .inputDataset(RemoteInferenceInputDataSet.builder().parameters(tmpParameters).build()).build());
            MLTaskResponse mlTaskResponse = client.execute(MLPredictionTaskAction.INSTANCE, request).actionGet(10_000);
            ModelTensorOutput tmpModelTensorOutput = (ModelTensorOutput) mlTaskResponse.getOutput();
            Object response = tmpModelTensorOutput.getMlModelOutputs().get(0).getMlModelTensors().get(0).getDataAsMap().get("response");

            String thought = "";
            if (response instanceof String) {
                thought = (String) response;
            } else if (response instanceof List) {
                Object value = ((List) response).get(0);
                thought = value instanceof String? (String)value : gson.toJson(value);
            }
            if (i == 0 && !thought.contains("Thought:")) {
                sessionMsgAnswerBuilder.append("Thought: ");
            }
            sessionMsgAnswerBuilder.append(thought);

            //check if verbose
            modelTensors.addAll(tmpModelTensorOutput.getMlModelOutputs());


            if (thought != null && thought.toLowerCase(Locale.ROOT).contains("final answer:")) {
                if (memory != null) {
                    memory.save(sessionId, ConversationMessage.conversationMessageBuilder().type("AI").content(thought).finalAnswer(true).build());
                }
                int indexOfFinalAnswer = thought.indexOf("Final Answer:");
                String finalAnswer = indexOfFinalAnswer >= 0? thought.substring(indexOfFinalAnswer + 13) : thought;
                if (finalAnswer.contains("\n\nQuestion:")) {
                    finalAnswer = finalAnswer.substring(0, finalAnswer.indexOf("\n\nQuestion:"));
                }
                if (finalAnswer.contains("\nHuman:")) {
                    finalAnswer = finalAnswer.substring(0, finalAnswer.indexOf("\nHuman:"));
                }
                cotModelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().name("response").result(finalAnswer).build())).build());

                List<ModelTensors> finalModelTensors = new ArrayList<>();
                finalModelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().name("response").dataAsMap(ImmutableMap.of("response", finalAnswer)).build())).build());
                return ModelTensorOutput.builder().mlModelOutputs(cotModelTensors).build();
            }
            List<String> actionRegexList = parameters.containsKey("cot.action_regex") ?
                    gson.fromJson(parameters.get("cot.action_regex"), List.class) :
                    Arrays.asList("Action:\\s*(\\w+)\\s*Action Input:\\s*(.*)", "action[:=]*\\s*([^\\n]+)\\s*action input[:=]*\\s*([^\\n]+)");

            String action = null;
            String actionInput = null;
            for (String actionRegex : actionRegexList) {
                Pattern pattern = Pattern.compile(actionRegex);
                Matcher matcher = pattern.matcher(thought);
                if (matcher.find()) {
                    action = matcher.group(1);
                    actionInput = matcher.group(2);
                }
                if (action != null && actionInput != null) {
                    action = action.trim();
                    actionInput = actionInput.trim();
                    break;
                }
            }

            String toolName = action;
            if (action != null) {
                for(String key : tools.keySet()){
                    if (action.toLowerCase().contains(key.toLowerCase())) {
                        toolName = key;
                    }
                }
            }
            action = toolName;

            String actionResult = null;
            if (action != null && tools.containsKey(action) && inputTools.contains(action)) {
                Map<String, String> toolParams = new HashMap<>();
                toolParams.put("input", actionInput);
                if (tools.get(action).validate(toolParams)) {
                    if (tools.get(action) instanceof MLModelTool) {
                        Map<String, String> llmToolTmpParameters = new HashMap<>();
                        llmToolTmpParameters.putAll(tmpParameters);
                        llmToolTmpParameters.put(QUESTION, actionInput);
                    }
                    actionResult = tools.get(action).run(toolParams);
                    modelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().dataAsMap(ImmutableMap.of("response", thought + "\nObservation: " + actionResult)).build())).build());
                } else {
                    actionResult = "Tool " + action + " can't work for input: " + actionInput;
                }

                thought = thought.replaceAll("Observation:.+\\n?", "").trim();
                scratchpadBuilder.append(thought).append("\nObservation: ").append(actionResult).append("\n\n");
                if (memory != null) {
//                    memory.save(sessionId, ConversationMessage.conversationMessageBuilder().type("AI").content(thought + "\nObservation: " + actionResult).finalAnswer(false).build());
                    memory.save(sessionId, ConversationMessage.conversationMessageBuilder().type("AI").content(thought).finalAnswer(false).build());
                    memory.save(sessionId, ConversationMessage.conversationMessageBuilder().type("Tool").content(action + " Observation: " + actionResult).finalAnswer(false).build());
                }

                tmpSubstitutor = new StringSubstitutor(ImmutableMap.of(SCRATCHPAD, scratchpadBuilder.toString()), "${parameters.", "}");
                newPrompt = tmpSubstitutor.replace(prompt);
                tmpParameters.put(PROMPT, newPrompt);
            } else {
                if (action != null) {
                    actionResult = "no access to this tool ";
                    scratchpadBuilder.append(thought).append("\nObservation: no access to this tool ").append(action).append("\n\n");
                } else {
                    log.info("tools not found, end this cot earlier");
                    String stopWhenNoToolFound = llm.getParameters().get("stop_when_no_tool_found");
                    if ("true".equalsIgnoreCase(stopWhenNoToolFound)) {
                        int indexOfFinalAnswer = thought.indexOf("Final Answer:");
                        String finalAnswer = indexOfFinalAnswer >= 0? thought.substring(indexOfFinalAnswer + 13) : thought;
                        if (finalAnswer.contains("\n\nQuestion:")) {
                            finalAnswer = finalAnswer.substring(0, finalAnswer.indexOf("\n\nQuestion:"));
                        }
                        if (finalAnswer.contains("\nHuman:")) {
                            finalAnswer = finalAnswer.substring(0, finalAnswer.indexOf("\nHuman:"));
                        }
                        cotModelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().name("response").result(finalAnswer).build())).build());
                        List<ModelTensors> finalModelTensors = new ArrayList<>();
                        finalModelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().name("response").dataAsMap(ImmutableMap.of("response", finalAnswer)).build())).build());
                        return ModelTensorOutput.builder().mlModelOutputs(cotModelTensors).build();
                    }
                    actionResult = "tool not found";
                    scratchpadBuilder.append(thought).append("\nObservation: tool not found").append("\n\n");
                }

                tmpSubstitutor = new StringSubstitutor(ImmutableMap.of(SCRATCHPAD, scratchpadBuilder.toString()), "${parameters.", "}");
                newPrompt = tmpSubstitutor.replace(prompt);
                tmpParameters.put(PROMPT, newPrompt);
            }
            sessionMsgAnswerBuilder.append("\nObservation: ").append(actionResult);
            cotModelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().name("response").result(sessionMsgAnswerBuilder.toString()).build())).build());
            if (i != 3 - 1 && parameters.containsKey("cot.step_interval_millis")) {
                Long interval = Long.parseLong(parameters.get("cot.step_interval_millis"));
                try {
                    Thread.sleep(interval);
                } catch (InterruptedException e) {
                    log.error("Failed to sleep", e);
                    throw new MLException(e);
                }
            }
        }
        List<ModelTensors> finalModelTensors = new ArrayList<>();
        List<ModelTensors> mlModelOutputs = modelTensorOutput.getMlModelOutputs();
        if (mlModelOutputs == null || mlModelOutputs.size() == 0) {
            throw new MLException("No output generated");
        }
        List<ModelTensor> mlModelTensors = mlModelOutputs.get(mlModelOutputs.size() - 1).getMlModelTensors();
        if (mlModelTensors == null || mlModelTensors.size() == 0) {
            throw new MLException("No valid response generated");
        }
        ModelTensor modelTensor = mlModelTensors.get(mlModelTensors.size() - 1);
        finalModelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(modelTensor)).build());
        return ModelTensorOutput.builder().mlModelOutputs(cotModelTensors).build();
    }

    private String addPrefixSuffixToPrompt(Map<String, String> parameters, String prompt) {
        Map<String, String> prefixMap = new HashMap<>();
        String prefix = parameters.containsKey(PROMPT_PREFIX) ? parameters.get(PROMPT_PREFIX) : "";
        String suffix = parameters.containsKey(PROMPT_SUFFIX) ? parameters.get(PROMPT_SUFFIX) : "";
        prefixMap.put(PROMPT_PREFIX, prefix);
        prefixMap.put(PROMPT_SUFFIX, suffix);
        StringSubstitutor substitutor = new StringSubstitutor(prefixMap, "${parameters.", "}");
        return substitutor.replace(prompt);
    }

    private String addToolsToPrompt(Map<String, Tool> tools, Map<String, String> parameters, List<String> inputTools, String prompt) {
        StringBuilder toolsBuilder = new StringBuilder();
        StringBuilder toolNamesBuilder = new StringBuilder();

        String toolsPrefix = Optional.ofNullable(parameters.get("agent.tools.prefix")).orElse("You have access to the following tools defined in <tools>: \n" + "<tools>\n");
        String toolsSuffix = Optional.ofNullable(parameters.get("agent.tools.suffix")).orElse("</tools>\n");
        String toolPrefix = Optional.ofNullable(parameters.get("agent.tools.tool.prefix")).orElse("<tool>\n");
        String toolSuffix = Optional.ofNullable(parameters.get("agent.tools.tool.suffix")).orElse("\n</tool>\n");
        toolsBuilder.append(toolsPrefix);
        for (String toolName : inputTools) {
            if (!tools.containsKey(toolName)) {
                throw new IllegalArgumentException("Tool ["+toolName+"] not registered for model");
            }
            toolsBuilder.append(toolPrefix).append(toolName).append(": ").append(tools.get(toolName).getDescription()).append(toolSuffix);
            toolNamesBuilder.append(toolName).append(", ");
        }
        toolsBuilder.append(toolsSuffix);
        Map<String, String> toolsPromptMap = new HashMap<>();
        toolsPromptMap.put(TOOL_DESCRIPTIONS, toolsBuilder.toString());
        toolsPromptMap.put(TOOL_NAMES, toolNamesBuilder.substring(0, toolNamesBuilder.length() - 1));

        if (parameters.containsKey(TOOL_DESCRIPTIONS)) {
            toolsPromptMap.put(TOOL_DESCRIPTIONS, parameters.get(TOOL_DESCRIPTIONS));
        }
        if (parameters.containsKey(TOOL_NAMES)) {
            toolsPromptMap.put(TOOL_NAMES, parameters.get(TOOL_NAMES));
        }
        StringSubstitutor substitutor = new StringSubstitutor(toolsPromptMap, "${parameters.", "}");
        return substitutor.replace(prompt);
    }

    private String addIndicesToPrompt(Map<String, String> parameters, String prompt) {
        Map<String, String> indicesMap = new HashMap<>();
        if (parameters.containsKey(OS_INDICES)) {
            String indices = parameters.get(OS_INDICES);
            List<String> indicesList = gson.fromJson(indices, List.class);
            StringBuilder indicesBuilder = new StringBuilder();
            String indicesPrefix = Optional.ofNullable(parameters.get("opensearch_indices.prefix")).orElse("You have access to the following OpenSearch Index defined in <opensearch_indexes>: \n" + "<opensearch_indexes>\n");
            String indicesSuffix = Optional.ofNullable(parameters.get("opensearch_indices.suffix")).orElse("</opensearch_indexes>\n");
            String indexPrefix = Optional.ofNullable(parameters.get("opensearch_indices.index.prefix")).orElse("<index>\n");
            String indexSuffix = Optional.ofNullable(parameters.get("opensearch_indices.index.suffix")).orElse("\n</index>\n");
            indicesBuilder.append(indicesPrefix);
            for (String e : indicesList) {
                indicesBuilder.append(indexPrefix).append(e).append(indexSuffix);
            }
            indicesBuilder.append(indicesSuffix);
            indicesMap.put(OS_INDICES, indicesBuilder.toString());
        } else {
            indicesMap.put(OS_INDICES, "");
        }
        StringSubstitutor substitutor = new StringSubstitutor(indicesMap, "${parameters.", "}");
        return substitutor.replace(prompt);
    }

    private String addExamplesToPrompt(Map<String, String> parameters, String prompt) {
        Map<String, String> examplesMap = new HashMap<>();
        if (parameters.containsKey(EXAMPLES)) {
            String examples = parameters.get(EXAMPLES);
            List<String> exampleList = gson.fromJson(examples, List.class);
            StringBuilder exampleBuilder = new StringBuilder();
            String examplesPrefix = Optional.ofNullable(parameters.get("examples.prefix")).orElse("You should follow and learn from examples defined in <examples>: \n" + "<examples>\n");
            String examplesSuffix = Optional.ofNullable(parameters.get("examples.suffix")).orElse("</examples>\n");
            exampleBuilder.append(examplesPrefix);

            String examplePrefix = Optional.ofNullable(parameters.get("examples.example.prefix")).orElse("<example>\n");
            String exampleSuffix = Optional.ofNullable(parameters.get("examples.example.suffix")).orElse("\n</example>\n");
            for (int i = 0; i< exampleList.size(); i++) {
                String example = exampleList.get(i);
                exampleBuilder.append(examplePrefix).append(example).append(exampleSuffix);
            }
            exampleBuilder.append(examplesSuffix);
            examplesMap.put(EXAMPLES, exampleBuilder.toString());
        } else {
            examplesMap.put(EXAMPLES, "");
        }
        StringSubstitutor substitutor = new StringSubstitutor(examplesMap, "${parameters.", "}");
        return substitutor.replace(prompt);
    }

    private String addChatHistoryToPrompt(Map<String, String> parameters, String prompt) {
        Map<String, String> chatHistoryMap = new HashMap<>();
        String chatHistory = parameters.containsKey(CHAT_HISTORY) ? parameters.get(CHAT_HISTORY) : "";
        chatHistoryMap.put(CHAT_HISTORY, chatHistory);
        parameters.put(CHAT_HISTORY, chatHistory);
        if (chatHistoryMap.size() > 0) {
            StringSubstitutor substitutor = new StringSubstitutor(chatHistoryMap, "${parameters.", "}");
            return substitutor.replace(prompt);
        }
        return prompt;
    }

    private String addContextToPrompt(Map<String, String> parameters, String prompt) {
        Map<String, String> contextMap = new HashMap<>();
        if (parameters.containsKey(CONTEXT)) {
            contextMap.put(CONTEXT, parameters.get(CONTEXT));
        } else {
            contextMap.put(CONTEXT, "");
        }
        parameters.put(CONTEXT, contextMap.get(CONTEXT));
        if (contextMap.size() > 0) {
            StringSubstitutor substitutor = new StringSubstitutor(contextMap, "${parameters.", "}");
            return substitutor.replace(prompt);
        }
        return prompt;
    }


}
