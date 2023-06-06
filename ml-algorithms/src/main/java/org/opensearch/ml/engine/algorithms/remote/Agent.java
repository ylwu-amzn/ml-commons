package org.opensearch.ml.engine.algorithms.remote;

import com.google.common.collect.ImmutableMap;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.text.StringSubstitutor;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.engine.tools.LanguageModelTool;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.opensearch.ml.engine.utils.ScriptUtils.gson;

@Log4j2
@Getter
public class Agent {

    private List<Tool> tools;
    private Map<String, Tool> toolsMap;
    private String defaultPrompt;

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

    public ModelTensorOutput run(Map<String, String> parameters, Function<Map<String, String>, ModelTensorOutput> executeDirectly) {
        Map<String, String> tmpParameters = new HashMap<>();
        tmpParameters.putAll(parameters);
        tmpParameters.put("stop", gson.toJson(new String[]{"\nObservation:", "\n\tObservation:"}));
        tmpParameters.put("stop_sequences", gson.toJson(new String[]{"\n\nHuman:", "\nObservation:", "\n\tObservation:","\nObservation", "\n\tObservation", "\n\nQuestion"}));

        Integer maxIterations = parameters.containsKey("max_iterations")? Integer.parseInt(parameters.get("max_iterations")) : 3;
        Boolean verbose = parameters.containsKey("verbose") ? Boolean.parseBoolean(parameters.get("verbose")) : false;
        String prompt = parameters.get("prompt");
        if (prompt == null) {
            prompt = defaultPrompt;
        }

        StringBuilder toolsBuilder = new StringBuilder();
        StringBuilder toolNamesBuilder = new StringBuilder();
        List<String> inputTools = new ArrayList<>();
        if (parameters.containsKey("tools")) {
            inputTools = gson.fromJson(parameters.get("tools"), List.class);
        } else {
            for (Tool t : tools) {
                inputTools.add(t.getName());
            }
        }

        for (String toolName : inputTools) {
            if (!toolsMap.containsKey(toolName)) {
                throw new IllegalArgumentException("Tool ["+toolName+"] not registered for model");
            }
            toolsBuilder.append(toolName).append(": ").append(toolsMap.get(toolName).getDescription()).append("\n");
            toolNamesBuilder.append(toolName).append(", ");
        }
        Map<String, String> toolsPromptMap = new HashMap<>();
        toolsPromptMap.put("tools", toolsBuilder.toString());
        toolsPromptMap.put("tool_names", toolNamesBuilder.substring(0, toolNamesBuilder.length() - 1));

        StringSubstitutor substitutor = new StringSubstitutor(toolsPromptMap);
        prompt = substitutor.replace(prompt);

        Map<String, String> indicesMap = new HashMap<>();
        if (parameters.containsKey("opensearch_indices")) {
            String indices = parameters.get("opensearch_indices");
            List<String> indicesList = gson.fromJson(indices, List.class);
            StringBuilder indicesBuilder = new StringBuilder("You have access to the following OpenSearch Index: \n");
            for (String e : indicesList) {
                indicesBuilder.append(e).append("\n");
            }
            indicesBuilder.append("\n");
            indicesMap.put("opensearch_indices", indicesBuilder.toString());
        } else {
            indicesMap.put("opensearch_indices", "");
        }
        substitutor = new StringSubstitutor(indicesMap);
        prompt = substitutor.replace(prompt);

        Map<String, String> examplesMap = new HashMap<>();
        if (parameters.containsKey("examples")) {
            String examples = parameters.get("examples");
            List<String> exampleList = gson.fromJson(examples, List.class);
            StringBuilder exampleBuilder = new StringBuilder("Examples: \n");
            for (String e : exampleList) {
                exampleBuilder.append(e).append("\n");
            }
            exampleBuilder.append("\n");
            examplesMap.put("examples", exampleBuilder.toString());
        } else {
            examplesMap.put("examples", "");
        }
        substitutor = new StringSubstitutor(examplesMap);
        prompt = substitutor.replace(prompt);

        Map<String, String> contextMap = new HashMap<>();
        if (parameters.containsKey("context")) {
            contextMap.put("context", parameters.get("context"));
        } else {
            contextMap.put("context", "");
        }
        if (parameters.containsKey("chat_history")) {
            contextMap.put("chat_history", parameters.get("chat_history"));
        } else {
            contextMap.put("chat_history", "");
        }
        if (contextMap.size() > 0) {
            substitutor = new StringSubstitutor(contextMap);
            prompt = substitutor.replace(prompt);
        }

        tmpParameters.put("prompt", prompt);

        List<ModelTensors> modelTensors = new ArrayList<>();
        ModelTensorOutput modelTensorOutput = new ModelTensorOutput(modelTensors);

        StringBuilder scratchpadBuilder = new StringBuilder();
        StringSubstitutor tmpSubstitutor = new StringSubstitutor(ImmutableMap.of("agent_scratchpad", scratchpadBuilder.toString()));
        String newPrompt = tmpSubstitutor.replace(prompt);
        tmpParameters.put("prompt", newPrompt);

        for (int i = 0; i<maxIterations; i++) {
            ModelTensorOutput tmpModelTensorOutput = executeDirectly.apply(tmpParameters);
            String response = (String) tmpModelTensorOutput.getMlModelOutputs().get(0).getMlModelTensors().get(0).getDataAsMap().get("response");
            if (response != null && response.toLowerCase(Locale.ROOT).contains("final answer:")) {
                modelTensors.addAll(tmpModelTensorOutput.getMlModelOutputs());

                String finalAnswer = response.substring(response.indexOf("Final Answer:") + 14, response.length());
                if (finalAnswer.contains("\n\nQuestion:")) {
                    finalAnswer = finalAnswer.substring(0, finalAnswer.indexOf("\n\nQuestion:"));
                }
                List<ModelTensors> finalModelTensors = new ArrayList<>();
                finalModelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().name("response").dataAsMap(ImmutableMap.of("response", finalAnswer)).build())).build());
                return verbose ? modelTensorOutput : new ModelTensorOutput(finalModelTensors);
            }
            Pattern pattern = Pattern.compile("Action:\\s*(\\w+)\\s*Action Input:\\s*(.*)");
            Matcher matcher = pattern.matcher(response);
            String action = null;
            String actionInput = null;
            if (matcher.find()) {
                action = matcher.group(1);
                actionInput = matcher.group(2);
            }

            if (action!= null && toolsMap.containsKey(action) && inputTools.contains(action)) {
                String result = null;
                if (toolsMap.get(action).validate(actionInput)) {
                    if (toolsMap.get(action) instanceof LanguageModelTool) {
                        ((LanguageModelTool) toolsMap.get(action)).setSupplier(() -> executeDirectly.apply(tmpParameters));
                    }
                    result = toolsMap.get(action).run(actionInput);
                }
                modelTensors.add(ModelTensors.builder().mlModelTensors(Arrays.asList(ModelTensor.builder().dataAsMap(ImmutableMap.of("response", response + "\nObservation: " + result)).build())).build());

                scratchpadBuilder.append(response).append("\nObservation: ").append(result).append("\n\n");

                tmpSubstitutor = new StringSubstitutor(ImmutableMap.of("agent_scratchpad", scratchpadBuilder.toString()));
                newPrompt = tmpSubstitutor.replace(prompt);
                tmpParameters.put("prompt", newPrompt);
            } else {
                scratchpadBuilder.append(response).append("\nObservation: no access to this tool ").append(action).append("\n\n");

                tmpSubstitutor = new StringSubstitutor(ImmutableMap.of("agent_scratchpad", scratchpadBuilder.toString()));
                newPrompt = tmpSubstitutor.replace(prompt);
                tmpParameters.put("prompt", newPrompt);
            }
        }
        tmpParameters.put("prompt", "Answer the following questions as best you can based on these context: \n\n"
                + scratchpadBuilder
                + contextMap.get("context")
                + contextMap.get("chat_history")
                + "\n\n Question: ${parameters.question}");
        ModelTensorOutput tmpModelTensorOutput = executeDirectly.apply(tmpParameters);
        modelTensors.addAll(tmpModelTensorOutput.getMlModelOutputs());
        return verbose ? modelTensorOutput : tmpModelTensorOutput;
    }
}
