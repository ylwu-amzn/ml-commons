package org.opensearch.ml.engine.algorithms.remote;

import com.google.common.collect.ImmutableMap;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.text.StringSubstitutor;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.engine.tools.LanguageModelTool;
import org.opensearch.ml.engine.tools.SearchIndexTool;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.opensearch.ml.engine.algorithms.remote.PromptTemplate.DEFAULT_LLM_PROMPT;
import static org.opensearch.ml.engine.utils.ScriptUtils.gson;

@Log4j2
@Getter
public class Agent {

    private List<Tool> tools;
    private Map<String, Tool> toolsMap;
    private String defaultPrompt;
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

    public Agent(List<Tool> tools, String defaultPrompt) {
        this.tools = tools;
        if (tools != null) {
            this.toolsMap = new HashMap<>();
            for (Tool tool : tools) {
                toolsMap.put(tool.getName(), tool);
            }
        }
        if (defaultPrompt != null) {
            this.defaultPrompt = defaultPrompt;
        } else {
            this.defaultPrompt = PromptTemplate.AGENT_TEMPLATE;
        }
    }

    public ModelTensorOutput run(Map<String, String> parameters,
                                 MLTask mlTask,
                                 Function<Map<String, String>, ModelTensorOutput> executeDirectly,
                                 Consumer<Map<String, Object>> saveSessionMessageConsumer) {
        String question = parameters.get(QUESTION);
        String taskId = mlTask.getTaskId();
        Map<String, String> tmpParameters = new HashMap<>();
        tmpParameters.putAll(parameters);
        if (!tmpParameters.containsKey("stop")) {
            tmpParameters.put("stop", gson.toJson(new String[]{"\nObservation:", "\n\tObservation:"}));
        }
        if (!tmpParameters.containsKey("stop_sequences")) {
            tmpParameters.put("stop_sequences", gson.toJson(new String[]{"\n\nHuman:", "\nObservation:", "\n\tObservation:","\nObservation", "\n\tObservation", "\n\nQuestion"}));
        }

        Integer maxIterations = parameters.containsKey("max_iterations")? Integer.parseInt(parameters.get("max_iterations")) : 3;
        if (maxIterations < 1) {
            throw new IllegalArgumentException("Max iterations must greater than 0");
        }
        Boolean verbose = parameters.containsKey("verbose") ? Boolean.parseBoolean(parameters.get("verbose")) : false;
        String prompt = parameters.get(PROMPT);
        if (prompt == null) {
            prompt = defaultPrompt;
        }

        List<String> inputTools = new ArrayList<>();
        if (parameters.containsKey(TOOLS)) {
            inputTools = gson.fromJson(parameters.get(TOOLS), List.class);
        } else {
            for (Tool t : tools) {
                inputTools.add(t.getName());
            }
        }

        prompt = addPrefixSuffixToPrompt(parameters, prompt);
        prompt = addToolsToPrompt(parameters, inputTools, prompt);
        prompt = addIndicesToPrompt(parameters, prompt);
        prompt = addExamplesToPrompt(parameters, prompt);
        prompt = addChatHistoryToPrompt(parameters, prompt);
        prompt = addContextToPrompt(parameters, prompt);

        tmpParameters.put(PROMPT, prompt);

        List<ModelTensors> modelTensors = new ArrayList<>();
        ModelTensorOutput modelTensorOutput = ModelTensorOutput.builder().mlModelOutputs(modelTensors).build();


        List<ModelTensors> cotModelTensors = new ArrayList<>();
        cotModelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().name(SESSION_ID).result(tmpParameters.get(SESSION_ID)).build())).build());

        StringBuilder scratchpadBuilder = new StringBuilder();
        StringSubstitutor tmpSubstitutor = new StringSubstitutor(ImmutableMap.of(SCRATCHPAD, scratchpadBuilder.toString()), "${parameters.", "}");
        String newPrompt = tmpSubstitutor.replace(prompt);
        tmpParameters.put(PROMPT, newPrompt);

        for (int i = 0; i<maxIterations; i++) {
            StringBuilder sessionMsgAnswerBuilder = new StringBuilder("");

            ModelTensorOutput tmpModelTensorOutput = executeDirectly.apply(tmpParameters);
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
            if (verbose) {
                modelTensors.addAll(tmpModelTensorOutput.getMlModelOutputs());
            }
            if (thought != null && thought.toLowerCase(Locale.ROOT).contains("final answer:")) {
                int indexOfFinalAnswer = thought.indexOf("Final Answer:");
                String finalAnswer = indexOfFinalAnswer >= 0? thought.substring(indexOfFinalAnswer + 13) : thought;
                if (finalAnswer.contains("\n\nQuestion:")) {
                    finalAnswer = finalAnswer.substring(0, finalAnswer.indexOf("\n\nQuestion:"));
                }
                if (finalAnswer.contains("\nHuman:")) {
                    finalAnswer = finalAnswer.substring(0, finalAnswer.indexOf("\nHuman:"));
                }
                cotModelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().name("response").result(finalAnswer).build())).build());
                if (saveSessionMessageConsumer != null) {
                    saveSessionMessageConsumer.accept(ImmutableMap.of(SESSION_ID, tmpParameters.get(SESSION_ID),
                            "question", question,
                            "answer", finalAnswer,
                            "final_answer", true,
                            "created_time", Instant.now().toEpochMilli(),
                            "task_id", taskId));
                }
                List<ModelTensors> finalModelTensors = new ArrayList<>();
                finalModelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().name("response").dataAsMap(ImmutableMap.of("response", finalAnswer)).build())).build());
                return verbose ? ModelTensorOutput.builder().mlModelOutputs(cotModelTensors).build() : ModelTensorOutput.builder().mlModelOutputs(finalModelTensors).build();
            }
            List<String> actionRegexList = parameters.containsKey("cot.action_regex") ?
                    gson.fromJson(parameters.get("cot.action_regex"), List.class) :
                    Arrays.asList("Action:\\s*(\\w+)\\s*Action Input:\\s*(.*)", "action[:=]*\\s*([^\\n]+)\\s*action input[:=]*\\s*([^\\n]+)");

            String action = null;
            String actionInput = null;
            for (String actionRegex : actionRegexList) {
                Pattern pattern = Pattern.compile(actionRegex);
                //Matcher matcher = pattern.matcher(thought.toLowerCase(Locale.ROOT));
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
                for(String key : toolsMap.keySet()){
                    if (action.toLowerCase().contains(key.toLowerCase())) {
                        toolName = key;
                    }
                }
            }
            action = toolName;

            String actionResult = null;
            if (action != null && toolsMap.containsKey(action) && inputTools.contains(action)) {
                Map<String, String> toolParams = new HashMap<>();
                if (toolsMap.get(action) instanceof SearchIndexTool) {
                    if (tmpParameters.containsKey("SearchIndexTool.doc_size")) {
                        toolParams.put("doc_size", tmpParameters.get("SearchIndexTool.doc_size"));
                    }
                }
                if (toolsMap.get(action).validate(actionInput, toolParams)) {
                    if (toolsMap.get(action) instanceof LanguageModelTool) {
                        Map<String, String> llmToolTmpParameters = new HashMap<>();
                        llmToolTmpParameters.putAll(tmpParameters);

                        String llmToolPrompt = tmpParameters.containsKey("LanguageModelTool.prompt")? tmpParameters.get("LanguageModelTool.prompt") : DEFAULT_LLM_PROMPT;
                        llmToolTmpParameters.put(PROMPT, llmToolPrompt);
                        llmToolTmpParameters.put(QUESTION, actionInput);
                        llmToolTmpParameters.put(LLM_TOOL_PROMPT_PREFIX, Optional.ofNullable(parameters.get(LLM_TOOL_PROMPT_PREFIX)).orElse(""));
                        llmToolTmpParameters.put(LLM_TOOL_PROMPT_SUFFIX, Optional.ofNullable(parameters.get(LLM_TOOL_PROMPT_SUFFIX)).orElse(""));
                        llmToolTmpParameters.put(SCRATCHPAD, scratchpadBuilder.toString());
                        if (parameters.containsKey("cot.step_interval_millis")) {
                            Long interval = Long.parseLong(parameters.get("cot.step_interval_millis"));
                            try {
                                Thread.sleep(interval);
                            } catch (InterruptedException e) {
                                log.error("Failed to sleep", e);
                                throw new MLException(e);
                            }
                        }
                        ((LanguageModelTool) toolsMap.get(action)).setSupplier(() -> executeDirectly.apply(llmToolTmpParameters));
                    }
                    actionResult = toolsMap.get(action).run(actionInput, toolParams);
                    modelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().dataAsMap(ImmutableMap.of("response", thought + "\nObservation: " + actionResult)).build())).build());
                } else {
                    actionResult = "Tool " + action + " can't work for input: " + actionInput;
                }

                thought = thought.replaceAll("Observation:.+\\n?", "").trim();
                scratchpadBuilder.append(thought).append("\nObservation: ").append(actionResult).append("\n\n");

                tmpSubstitutor = new StringSubstitutor(ImmutableMap.of(SCRATCHPAD, scratchpadBuilder.toString()), "${parameters.", "}");
                newPrompt = tmpSubstitutor.replace(prompt);
                tmpParameters.put(PROMPT, newPrompt);
            } else {
                if (action != null) {
                    actionResult = "no access to this tool ";
                    scratchpadBuilder.append(thought).append("\nObservation: no access to this tool ").append(action).append("\n\n");
                } else {
                    log.info("tools not found, end this cot earlier");
                    String stopEarly = parameters.get("cot.stop_when_no_tool_found");
                    if ("true".equalsIgnoreCase(stopEarly)) {
                        String finalAnswer = thought.substring(thought.indexOf("Final Answer:") + 14, thought.length());
                        if (finalAnswer.contains("\n\nQuestion:")) {
                            finalAnswer = finalAnswer.substring(0, finalAnswer.indexOf("\n\nQuestion:"));
                        }
                        if (finalAnswer.contains("\nHuman:")) {
                            finalAnswer = finalAnswer.substring(0, finalAnswer.indexOf("\nHuman:"));
                        }
                        cotModelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().name("response").result(finalAnswer).build())).build());
                        if (saveSessionMessageConsumer != null) {
                            saveSessionMessageConsumer.accept(ImmutableMap.of(SESSION_ID, tmpParameters.get(SESSION_ID),
                                    "question", question,
                                    "answer", finalAnswer,
                                    "final_answer", true,
                                    "created_time", Instant.now().toEpochMilli(),
                                    "task_id", taskId));
                        }
                        List<ModelTensors> finalModelTensors = new ArrayList<>();
                        finalModelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().name("response").dataAsMap(ImmutableMap.of("response", finalAnswer)).build())).build());
                        return verbose ? ModelTensorOutput.builder().mlModelOutputs(cotModelTensors).build() : ModelTensorOutput.builder().mlModelOutputs(finalModelTensors).build();
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
            if (saveSessionMessageConsumer != null && (verbose || i == maxIterations - 1)) {
                saveSessionMessageConsumer.accept(ImmutableMap.of(SESSION_ID, tmpParameters.get(SESSION_ID),
                        "question", question,
                        "answer", sessionMsgAnswerBuilder.toString(),
                        "final_answer", i == maxIterations - 1,
                        "created_time", Instant.now().toEpochMilli(),
                        "task_id", taskId));
            }
            if (i != maxIterations - 1 && parameters.containsKey("cot.step_interval_millis")) {
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
        return verbose ? ModelTensorOutput.builder().mlModelOutputs(cotModelTensors).build() : ModelTensorOutput.builder().mlModelOutputs(finalModelTensors).build();
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
    private String addToolsToPrompt(Map<String, String> parameters, List<String> inputTools, String prompt) {
        StringBuilder toolsBuilder = new StringBuilder();
        StringBuilder toolNamesBuilder = new StringBuilder();

        for (String toolName : inputTools) {
            if (!toolsMap.containsKey(toolName)) {
                throw new IllegalArgumentException("Tool ["+toolName+"] not registered for model");
            }
            toolsBuilder.append(toolName).append(": ").append(toolsMap.get(toolName).getDescription()).append("\n");
            toolNamesBuilder.append(toolName).append(", ");
        }
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
            StringBuilder indicesBuilder = new StringBuilder("You have access to the following OpenSearch Index: \n");
            for (String e : indicesList) {
                indicesBuilder.append(e).append("\n");
            }
            indicesBuilder.append("\nEnd of OpenSearch Index\n");
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
            StringBuilder exampleBuilder = new StringBuilder("\nExamples: \n\n");
            for (int i = 0; i< exampleList.size(); i++) {
                String example = exampleList.get(i);
                exampleBuilder.append("Example ").append(i + 1).append(":\n").append(example).append("\n");
            }
            exampleBuilder.append("\nEnd of Examples\n");
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
