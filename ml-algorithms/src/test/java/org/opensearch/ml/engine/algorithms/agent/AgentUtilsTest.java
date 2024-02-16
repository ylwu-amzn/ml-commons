/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.agent;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.when;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.PROMPT_PREFIX;
import static org.opensearch.ml.engine.algorithms.agent.AgentUtils.PROMPT_SUFFIX;
import static org.opensearch.ml.engine.algorithms.agent.MLChatAgentRunner.CHAT_HISTORY;
import static org.opensearch.ml.engine.algorithms.agent.MLChatAgentRunner.CONTEXT;
import static org.opensearch.ml.engine.algorithms.agent.MLChatAgentRunner.EXAMPLES;
import static org.opensearch.ml.engine.algorithms.agent.MLChatAgentRunner.OS_INDICES;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.gson.Gson;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.utils.StringUtils;

public class AgentUtilsTest {

    @Mock
    private Tool tool1, tool2;

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);

    }

    @Test
    public void testAddIndicesToPrompt_WithIndices() {
        String initialPrompt = "initial prompt ${parameters.opensearch_indices}";
        Map<String, String> parameters = new HashMap<>();
        parameters.put(OS_INDICES, "[\"index1\", \"index2\"]");

        String expected =
            "initial prompt You have access to the following OpenSearch Index defined in <opensearch_indexes>: \n<opensearch_indexes>\n"
                + "<index>\nindex1\n</index>\n<index>\nindex2\n</index>\n</opensearch_indexes>\n";

        String result = AgentUtils.addIndicesToPrompt(parameters, initialPrompt);
        assertEquals(expected, result);
    }

    @Test
    public void testAddIndicesToPrompt_WithoutIndices() {
        String prompt = "initial prompt";
        Map<String, String> parameters = new HashMap<>();

        String expected = "initial prompt";

        String result = AgentUtils.addIndicesToPrompt(parameters, prompt);
        assertEquals(expected, result);
    }

    @Test
    public void testAddIndicesToPrompt_WithCustomPrefixSuffix() {
        String initialPrompt = "initial prompt ${parameters.opensearch_indices}";
        Map<String, String> parameters = new HashMap<>();
        parameters.put(OS_INDICES, "[\"index1\", \"index2\"]");
        parameters.put("opensearch_indices.prefix", "Custom Prefix\n");
        parameters.put("opensearch_indices.suffix", "\nCustom Suffix");
        parameters.put("opensearch_indices.index.prefix", "Index: ");
        parameters.put("opensearch_indices.index.suffix", "; ");

        String expected = "initial prompt Custom Prefix\nIndex: index1; Index: index2; \nCustom Suffix";

        String result = AgentUtils.addIndicesToPrompt(parameters, initialPrompt);
        assertEquals(expected, result);
    }

    @Test
    public void testAddExamplesToPrompt_WithExamples() {
        // Setup
        String initialPrompt = "initial prompt ${parameters.examples}";
        Map<String, String> parameters = new HashMap<>();
        parameters.put(EXAMPLES, "[\"Example 1\", \"Example 2\"]");

        // Expected output
        String expectedPrompt = "initial prompt EXAMPLES\n--------\n"
            + "You should follow and learn from examples defined in <examples>: \n"
            + "<examples>\n"
            + "<example>\nExample 1\n</example>\n"
            + "<example>\nExample 2\n</example>\n"
            + "</examples>\n";

        // Call the method under test
        String actualPrompt = AgentUtils.addExamplesToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddExamplesToPrompt_WithoutExamples() {
        // Setup
        String initialPrompt = "initial prompt ${parameters.examples}";
        Map<String, String> parameters = new HashMap<>();

        // Expected output (should remain unchanged)
        String expectedPrompt = "initial prompt ";

        // Call the method under test
        String actualPrompt = AgentUtils.addExamplesToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddPrefixSuffixToPrompt_WithPrefixSuffix() {
        // Setup
        String initialPrompt = "initial prompt ${parameters.prompt.prefix} main content ${parameters.prompt.suffix}";
        Map<String, String> parameters = new HashMap<>();
        parameters.put(PROMPT_PREFIX, "Prefix: ");
        parameters.put(PROMPT_SUFFIX, " :Suffix");

        // Expected output
        String expectedPrompt = "initial prompt Prefix:  main content  :Suffix";

        // Call the method under test
        String actualPrompt = AgentUtils.addPrefixSuffixToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddPrefixSuffixToPrompt_WithoutPrefixSuffix() {
        // Setup
        String initialPrompt = "initial prompt ${parameters.prompt.prefix} main content ${parameters.prompt.suffix}";
        Map<String, String> parameters = new HashMap<>();

        // Expected output (should remain unchanged)
        String expectedPrompt = "initial prompt  main content ";

        // Call the method under test
        String actualPrompt = AgentUtils.addPrefixSuffixToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddToolsToPrompt_WithDescriptions() {
        // Setup
        Map<String, Tool> tools = new HashMap<>();
        tools.put("Tool1", tool1);
        tools.put("Tool2", tool2);
        when(tool1.getDescription()).thenReturn("Description of Tool1");
        when(tool2.getDescription()).thenReturn("Description of Tool2");

        List<String> inputTools = Arrays.asList("Tool1", "Tool2");
        String initialPrompt = "initial prompt ${parameters.tool_descriptions} and ${parameters.tool_names}";

        // Expected output
        String expectedPrompt = "initial prompt You have access to the following tools defined in <tools>: \n"
            + "<tools>\n<tool>\nTool1: Description of Tool1\n</tool>\n"
            + "<tool>\nTool2: Description of Tool2\n</tool>\n</tools>\n and Tool1, Tool2,";

        // Call the method under test
        String actualPrompt = AgentUtils.addToolsToPrompt(tools, new HashMap<>(), inputTools, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddToolsToPrompt_ToolNotRegistered() {
        // Setup
        Map<String, Tool> tools = new HashMap<>();
        tools.put("Tool1", tool1);
        List<String> inputTools = Arrays.asList("Tool1", "UnregisteredTool");
        String initialPrompt = "initial prompt ${parameters.tool_descriptions}";

        // Assert
        assertThrows(IllegalArgumentException.class, () -> AgentUtils.addToolsToPrompt(tools, new HashMap<>(), inputTools, initialPrompt));
    }

    @Test
    public void testAddChatHistoryToPrompt_WithChatHistory() {
        // Setup
        Map<String, String> parameters = new HashMap<>();
        parameters.put(CHAT_HISTORY, "Previous chat history here.");
        String initialPrompt = "initial prompt ${parameters.chat_history}";

        // Expected output
        String expectedPrompt = "initial prompt Previous chat history here.";

        // Call the method under test
        String actualPrompt = AgentUtils.addChatHistoryToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddChatHistoryToPrompt_NoChatHistory() {
        // Setup
        Map<String, String> parameters = new HashMap<>();
        String initialPrompt = "initial prompt ${parameters.chat_history}";

        // Expected output (no change from initial prompt)
        String expectedPrompt = "initial prompt ";

        // Call the method under test
        String actualPrompt = AgentUtils.addChatHistoryToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddContextToPrompt_WithContext() {
        // Setup
        Map<String, String> parameters = new HashMap<>();
        parameters.put(CONTEXT, "Contextual information here.");
        String initialPrompt = "initial prompt ${parameters.context}";

        // Expected output
        String expectedPrompt = "initial prompt Contextual information here.";

        // Call the method under test
        String actualPrompt = AgentUtils.addContextToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testAddContextToPrompt_NoContext() {
        // Setup
        Map<String, String> parameters = new HashMap<>();
        String initialPrompt = "initial prompt ${parameters.context}";

        // Expected output (no change from initial prompt)
        String expectedPrompt = "initial prompt ";

        // Call the method under test
        String actualPrompt = AgentUtils.addContextToPrompt(parameters, initialPrompt);

        // Assert
        assertEquals(expectedPrompt, actualPrompt);
    }

    @Test
    public void testExtractModelResponseJsonWithInvalidModelOutput() {
        String text = "invalid output";
        assertThrows(IllegalArgumentException.class, () -> AgentUtils.extractModelResponseJson(text));
    }

    @Test
    public void testExtractModelResponseJsonWithValidModelOutput() {
        String text =
            "This is the model response\n```json\n{\"thought\":\"use CatIndexTool to get index first\",\"action\":\"CatIndexTool\"}```";
        String responseJson = AgentUtils.extractModelResponseJson(text);
        assertEquals("{\"thought\":\"use CatIndexTool to get index first\",\"action\":\"CatIndexTool\"}", responseJson);
    }

    @Test
    public void testExtractModelResponseJson_ThoughtFinalAnswer() {
        String text =
            "---------------------\n{\n  \"thought\": \"Unfortunately the tools did not provide the weather forecast directly. Let me check online sources:\",\n  \"final_answer\": \"After checking online weather forecasts, it looks like tomorrow will be sunny with a high of 25 degrees Celsius.\"\n}";
        String result = AgentUtils.extractModelResponseJson(text);
        String expectedResult = "{\n"
            + "  \"thought\": \"Unfortunately the tools did not provide the weather forecast directly. Let me check online sources:\",\n"
            + "  \"final_answer\": \"After checking online weather forecasts, it looks like tomorrow will be sunny with a high of 25 degrees Celsius.\"\n"
            + "}";
        System.out.println(result);
        Assert.assertEquals(expectedResult, result);
    }

    @Test
    public void testExtractModelResponseJson_ThoughtFinalAnswerJsonBlock() {
        String text =
            "---------------------```json\n{\n  \"thought\": \"Unfortunately the tools did not provide the weather forecast directly. Let me check online sources:\",\n  \"final_answer\": \"After checking online weather forecasts, it looks like tomorrow will be sunny with a high of 25 degrees Celsius.\"\n}\n```";
        String result = AgentUtils.extractModelResponseJson(text);
        String expectedResult = "{\n"
            + "  \"thought\": \"Unfortunately the tools did not provide the weather forecast directly. Let me check online sources:\",\n"
            + "  \"final_answer\": \"After checking online weather forecasts, it looks like tomorrow will be sunny with a high of 25 degrees Celsius.\"\n"
            + "}";
        System.out.println(result);
        Assert.assertEquals(expectedResult, result);
    }

    @Test
    public void testExtractModelResponseJson_ThoughtActionInput() {
        String text =
            "---------------------\n{\n  \"thought\": \"Let me search our index to find population projections\", \n  \"action\": \"VectorDBTool\",\n  \"action_input\": \"Seattle population projection 2023\"\n}";
        String result = AgentUtils.extractModelResponseJson(text);
        String expectedResult = "{\n"
            + "  \"thought\": \"Let me search our index to find population projections\", \n"
            + "  \"action\": \"VectorDBTool\",\n"
            + "  \"action_input\": \"Seattle population projection 2023\"\n"
            + "}";
        System.out.println(result);
        Assert.assertEquals(expectedResult, result);
    }

    @Test
    public void test1() {
        String text = "```json\n{\n    \"thought\": \"Now I know the final answer\",\n    \"final_answer\": \"Sure, here is a simple Python script that reads a local file line by line and ingests it into an OpenSearch cluster using the _bulk API. Please replace 'localhost', 'port', 'index_name', and 'file_path' with your actual values.\n\n```python\nimport json\nimport requests\n\ndef read_file(file_path):\n    with open(file_path, 'r') as file:\n        for line in file:\n            yield line\n\ndef bulk_insert(data, index_name):\n    url = 'http://localhost:port/{}/_bulk'.format(index_name)\n    headers = {'Content-Type': 'application/x-ndjson'}\n    response = requests.post(url, headers=headers, data=data)\n    return response.json()\n\ndef prepare_bulk_data(file_path):\n    bulk_data = ''\n    for line in read_file(file_path):\n        index = { 'index' : {} }\n        bulk_data += json.dumps(index) + '\\\\n'\n        bulk_data += line + '\\\\n'\n    return bulk_data\n\ndef ingest_data(file_path, index_name):\n    bulk_data = prepare_bulk_data(file_path)\n    response = bulk_insert(bulk_data, index_name)\n    print(response)\n\ningest_data('file_path', 'index_name')\n```\nThis script first reads the file line by line, then prepares the bulk data in the format required by the _bulk API, and finally sends a POST request to the OpenSearch cluster. Please note that error handling and exception management are not included in this script, so you may want to add those according to your needs.\"\n}\n```";
        //String result = AgentUtils.extractModelResponseJson(text, List.of("\\{\\s*(\"(thought|action|action_input|final_answer)\"\\s*:\\s*\"[^\"]*\"\\s*,?\\s*)+\\}"));
//        String result = AgentUtils.extractModelResponseJson(text, List.of("\\{\\s*(\"(thought|action|action_input|final_answer)\"\\s*:\\s*\"[^\"]*\"\\s*,?\\s*)+\\}"));
        String result = AgentUtils.extractModelResponseJson(text);
        Gson gson = new Gson();
        System.out.println(gson.toJson(result));
        if (result.contains("\"final_answer\"")) {
            String pattern = "\"final_answer\"\\s*:\\s*\"(.*?)\"";
            Pattern jsonBlockPattern = Pattern.compile(pattern);
            Matcher jsonBlockMatcher = jsonBlockPattern.matcher(result);
            while (jsonBlockMatcher.find()) {
                String group = jsonBlockMatcher.group(1);
                System.out.println("Match found: " + group);
            }
        }
        Assert.assertTrue(StringUtils.isJson(result));;
    }

    @Test
    public void test2 () {
        String result = "{\n    \"thought\": \"Now I know the final answer\",\n    \"final_answer\": \"Sure, here\\\" is a simple Python script that reads a local file line by line and ingests it into an OpenSearch cluster using the _bulk API. Please replace \u0027localhost\u0027, \u0027port\u0027, \u0027index_name\u0027, and \u0027file_path\u0027 with your actual values.\n\n```python\nimport json\nimport requests\n\ndef read_file(file_path):\n    with open(file_path, \u0027r\u0027) as file:\n        for line in file:\n            yield line\n\ndef bulk_insert(data, index_name):\n    url \u003d \u0027http://localhost:port/{}/_bulk\u0027.format(index_name)\n    headers \u003d {\u0027Content-Type\u0027: \u0027application/x-ndjson\u0027}\n    response \u003d requests.post(url, headers\u003dheaders, data\u003ddata)\n    return response.json()\n\ndef prepare_bulk_data(file_path):\n    bulk_data \u003d \u0027\u0027\n    for line in read_file(file_path):\n        index \u003d { \u0027index\u0027 : {} }\n        bulk_data +\u003d json.dumps(index) + \u0027\\\\n\u0027\n        bulk_data +\u003d line + \u0027\\\\n\u0027\n    return bulk_data\n\ndef ingest_data(file_path, index_name):\n    bulk_data \u003d prepare_bulk_data(file_path)\n    response \u003d bulk_insert(bulk_data, index_name)\n    print(response)\n\ningest_data(\u0027file_path\u0027, \u0027index_name\u0027)\n```\nThis script first reads the file line by line, then prepares the bulk data in the format required by the _bulk API, and finally sends a POST request to the OpenSearch cluster. Please note that error handling and exception management are not included in this script, so you may want to add those according to your needs.\"\n}";
        Gson gson = new Gson();
        System.out.println(gson.toJson(result));
        String group=null;
        if (result.contains("\"final_answer\"")) {
            String pattern = "\"final_answer\"\\s*:\\s*\"(.*?)\"";
            Pattern jsonBlockPattern = Pattern.compile(pattern, Pattern.DOTALL); // Add Pattern.DOTALL to match across newlines
            Matcher jsonBlockMatcher = jsonBlockPattern.matcher(result);
            while (jsonBlockMatcher.find()) {
                group = jsonBlockMatcher.group(1);
                System.out.println("Match found: " + group);
            }
        }
        Assert.assertNotNull(group);
    }

    @Test
    public void testExtractThought () {
        String finalAnswer = "{\n    \"thought\": \"Now I \n know the\" final answer \",\n    \"final_answer\": \"Sure, here\\\" is a simple Python script that reads a local file line by line and ingests it into an OpenSearch cluster using the _bulk API. Please replace \u0027localhost\u0027, \u0027port\u0027, \u0027index_name\u0027, and \u0027file_path\u0027 with your actual values.\n\n```python\nimport json\nimport requests\n\ndef read_file(file_path):\n    with open(file_path, \u0027r\u0027) as file:\n        for line in file:\n            yield line\n\ndef bulk_insert(data, index_name):\n    url \u003d \u0027http://localhost:port/{}/_bulk\u0027.format(index_name)\n    headers \u003d {\u0027Content-Type\u0027: \u0027application/x-ndjson\u0027}\n    response \u003d requests.post(url, headers\u003dheaders, data\u003ddata)\n    return response.json()\n\ndef prepare_bulk_data(file_path):\n    bulk_data \u003d \u0027\u0027\n    for line in read_file(file_path):\n        index \u003d { \u0027index\u0027 : {} }\n        bulk_data +\u003d json.dumps(index) + \u0027\\\\n\u0027\n        bulk_data +\u003d line + \u0027\\\\n\u0027\n    return bulk_data\n\ndef ingest_data(file_path, index_name):\n    bulk_data \u003d prepare_bulk_data(file_path)\n    response \u003d bulk_insert(bulk_data, index_name)\n    print(response)\n\ningest_data(\u0027file_path\u0027, \u0027index_name\u0027)\n```\nThis script first reads the file line by line, then prepares the bulk data in the format required by the _bulk API, and finally sends a POST request to the OpenSearch cluster. Please note that error handling and exception management are not included in this script, so you may want to add those according to your needs.\"\n}";
        String thoughtResult = AgentUtils.extractThought(finalAnswer);
        Assert.assertEquals("Now I \n know the\" final answer ", thoughtResult);

        String action = "{\n    \"thought\": \"Let's run \n some\" tool get more data \",\n   \n \"action\": \"Ok now let's run CatIndexTool\"\n}";
        thoughtResult = AgentUtils.extractThought(action);
        Assert.assertEquals("Let's run \n some\" tool get more data ", thoughtResult);
    }

    @Test
    public void testExtractFinalAnswer () {
        String text = "{\n    \"thought\": \"Now I \n know the\" final answer \",\n    \"final_answer\": \"Sure, here\\\" is { a simple Python } script that reads a local file line by line and ingests it into an OpenSearch cluster using the _bulk API. Please replace \u0027localhost\u0027, \u0027port\u0027, \u0027index_name\u0027, and \u0027file_path\u0027 with your actual values.\n\n```python\nimport json\nimport requests\n\ndef read_file(file_path):\n    with open(file_path, \u0027r\u0027) as file:\n        for line in file:\n            yield line\n\ndef bulk_insert(data, index_name):\n    url \u003d \u0027http://localhost:port/{}/_bulk\u0027.format(index_name)\n    headers \u003d {\u0027Content-Type\u0027: \u0027application/x-ndjson\u0027}\n    response \u003d requests.post(url, headers\u003dheaders, data\u003ddata)\n    return response.json()\n\ndef prepare_bulk_data(file_path):\n    bulk_data \u003d \u0027\u0027\n    for line in read_file(file_path):\n        index \u003d { \u0027index\u0027 : {} }\n        bulk_data +\u003d json.dumps(index) + \u0027\\\\n\u0027\n        bulk_data +\u003d line + \u0027\\\\n\u0027\n    return bulk_data\n\ndef ingest_data(file_path, index_name):\n    bulk_data \u003d prepare_bulk_data(file_path)\n    response \u003d bulk_insert(bulk_data, index_name)\n    print(response)\n\ningest_data(\u0027file_path\u0027, \u0027index_name\u0027)\n```\nThis script first reads the file line by line, then prepares the bulk data in the format required by the _bulk API, and finally sends a POST request to the OpenSearch cluster. Please note that error handling and exception management are not included in this script, so you may want to add those according to your needs.\"\n}";
        String result = AgentUtils.extractFinalAnswer(text);
        Assert.assertEquals("Sure, here\\\" is { a simple Python } script that reads a local file line by line and ingests it into an OpenSearch cluster using the _bulk API. Please replace \u0027localhost\u0027, \u0027port\u0027, \u0027index_name\u0027, and \u0027file_path\u0027 with your actual values.\n\n```python\nimport json\nimport requests\n\ndef read_file(file_path):\n    with open(file_path, \u0027r\u0027) as file:\n        for line in file:\n            yield line\n\ndef bulk_insert(data, index_name):\n    url \u003d \u0027http://localhost:port/{}/_bulk\u0027.format(index_name)\n    headers \u003d {\u0027Content-Type\u0027: \u0027application/x-ndjson\u0027}\n    response \u003d requests.post(url, headers\u003dheaders, data\u003ddata)\n    return response.json()\n\ndef prepare_bulk_data(file_path):\n    bulk_data \u003d \u0027\u0027\n    for line in read_file(file_path):\n        index \u003d { \u0027index\u0027 : {} }\n        bulk_data +\u003d json.dumps(index) + \u0027\\\\n\u0027\n        bulk_data +\u003d line + \u0027\\\\n\u0027\n    return bulk_data\n\ndef ingest_data(file_path, index_name):\n    bulk_data \u003d prepare_bulk_data(file_path)\n    response \u003d bulk_insert(bulk_data, index_name)\n    print(response)\n\ningest_data(\u0027file_path\u0027, \u0027index_name\u0027)\n```\nThis script first reads the file line by line, then prepares the bulk data in the format required by the _bulk API, and finally sends a POST request to the OpenSearch cluster. Please note that error handling and exception management are not included in this script, so you may want to add those according to your needs.\"\n}", result);
    }

    @Test
    public void testExtractFinalAnswer_NoMatch() {
        String text = "{\n    \"thought\": \"Let's run \n some\" tool get more data \",\n   \n \"action\": \"Ok now let's run \n{ CatIndexTool }\"\n}";
        String result = AgentUtils.extractFinalAnswer(text);
        Assert.assertNull(result);
    }

    @Test
    public void testExtractAction_NoActionInput() {
        String text = "{\n    \"thought\": \"Let's run \n some\" tool get more data \",\n   \n \"action\": \"Ok now let's run \n{ CatIndexTool }\"\n}";
        String result = AgentUtils.extractAction(text);
        Assert.assertEquals("Ok now let\u0027s run \n{ CatIndexTool }\"\n}", result);
    }

    @Test
    public void testExtractAction_ActionInput() {
        String text = "{\n    \"thought\": \"Let's run \n some\" tool get more data \",\n   \n \"action\": \"Ok now let's run \n{ CatIndexTool }\", \n \"action_input\": 123 \n}";
        String result = AgentUtils.extractAction(text);
        Assert.assertEquals("Ok now let's run \n{ CatIndexTool }\", \n ", result);
    }

    @Test
    public void testExtractActionInput() {
//        String text = "{\n    \"thought\": \"Let's run \n some\" tool get more data \",\n   \n \"action\": \"Ok now let's run \n{ CatIndexTool }\", \n \"action_input\": 123 \n}";
//        String text = "{\n    \"thought\": \"I need to use the stock_price_data_knowledge_base tool to get the stock price data for Amazon from May 2023 to Jan 2024.\",\n    \"action\": \"stock_price_data_knowledge_base\",\n    \"action_input\": \"{\\\"company_name\\\": \\\"Amazon\\\", \\\"start_date\\\": \\\"2023-05-01\\\", \\\"end_date\\\": \\\"2024-01-31\\\"}\"\n}";
//        String text = "{\n    \"thought\": \"I need to use the stock_price_data_knowledge_base tool to get the stock price data for Amazon from May 2023 to Jan 2024.\",\n    \"action\": \"stock_price_data_knowledge_base\",\n    \"action_input\": \"{\\\"company_name\\\": \\\"Amazon\\\\\\\\\"inc \\\", \\\"start_date\\\": \\\"2023-05-01\\\", \\\"end_date\\\": \\\"2024-01-31\\\"}\"\n}";
        String text = "{\n    \"thought\": \"I need to use the stock_price_data_knowledge_base tool to get the stock price data for Amazon from May 2023 to Jan 2024.\",\n    \"action\": \"stock_price_data_knowledge_base\",\n    \"action_input\": \"Amazon\"\n}";
        String result = AgentUtils.extractActionInput(text);
        Assert.assertEquals("Ok now let's run \n{ CatIndexTool }\", \n ", result);
    }

    @Test
    public void testExtractActionInput2() {
//        Gson gson = new Gson();
//        gson.fromJson("{\\\"company_name\\\": \\\"Amazon\\\", \\\"start_date\\\": \\\"2023-05-01\\\", \\\"end_date\\\": \\\"2024-01-31\\\"}")
    }
}
