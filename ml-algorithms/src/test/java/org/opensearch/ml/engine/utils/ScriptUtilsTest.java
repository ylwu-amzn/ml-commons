/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.utils;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.opensearch.ml.engine.utils.ScriptUtils.executePostProcessFunction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.gson.Gson;
import com.jayway.jsonpath.JsonPath;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.ingest.TestTemplateService;
import org.opensearch.ml.common.connector.MLPostProcessFunction;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.utils.StringUtils;
import org.opensearch.script.ScriptService;

public class ScriptUtilsTest {

    @Mock
    ScriptService scriptService;

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
        when(scriptService.compile(any(), any())).then(invocation -> new TestTemplateService.MockTemplateScript.Factory("test result"));
    }

    @Test
    public void test_executePreprocessFunction() {
        Optional<String> resultOpt = ScriptUtils
            .executePreprocessFunction(scriptService, "any function", Collections.singletonList("any input"));
        assertEquals("test result", resultOpt.get());
    }

    @Test
    public void test_executeBuildInPostProcessFunction() {
        List<List<Float>> input = Arrays.asList(Arrays.asList(1.0f, 2.0f), Arrays.asList(3.0f, 4.0f));
        List<ModelTensor> modelTensors = MLPostProcessFunction.get(MLPostProcessFunction.DEFAULT_EMBEDDING).apply(input);
        assertNotNull(modelTensors);
        assertEquals(2, modelTensors.size());
    }

    @Test
    public void test_executePostProcessFunction() {
        when(scriptService.compile(any(), any())).then(invocation -> new TestTemplateService.MockTemplateScript.Factory("{\"result\": \"test result\"}"));
        Optional<String> resultOpt = executePostProcessFunction(scriptService, "any function", "{\"result\": \"test result\"}");
        assertEquals("{\"result\": \"test result\"}", resultOpt.get());
    }

    @Test
    public void t2() {
        String s = "\n    def name = 'response';\n    def result = process_function.json_path_filter($.text);\n    def json = \"{\" +\n          '\"name\": \"' + name + '\",' +\n          '\"dataAsMap\": { \"completion\":  \"' + result +\n          '\"}}';\n    return json;\n    ";
        String resultJson = "{\"text\": \"hello \\n \\\" aaa\"}";
        executePostProcessFunction(null, s, resultJson);
    }
    @Test
    public void test1() {
        String resultJson = "[\"abc\n123\", 111]";
        Map<String, Object> result = StringUtils.fromJson(resultJson, "result");
        System.out.println(result);

        String temp = "def r = process_function.escape(process_function.json_path_filter($.[0])); \ndef r2 = process_function.escape(process_function.json_path_filter($.[1])); \n\ndef r3 = process_function.escape(process_function.json_path_filter($.[1])); \n return \"result\": + r + r2; ";

        List<String> strings = extractStrings(temp);

        Gson gson = new Gson();
        for (String s : strings) {
            Object filteredOutput = JsonPath.read(resultJson, s);
//            String escape = escape(filteredOutput);
            String escape = gson.toJson(filteredOutput);
            temp = temp.replace("process_function.json_path_filter("+s+")", escape);
            System.out.println(temp);
        }

        Object filteredOutput = JsonPath.read(resultJson, "$.[0]");
        System.out.println(filteredOutput);

    }

    String escape(Object obj) {Gson gson = new Gson();
        if (obj instanceof String) {
            return gson.toJson(obj);
        }else {
            return gson.toJson(gson.toJson(obj));
        }
    }



    List<String> extractStrings(String input) {
        List<String> extractedStrings = new ArrayList<>();

        // Define the pattern to match strings following the specified pattern
        Pattern pattern = Pattern.compile("process_function\\.json_path_filter\\((.*?)\\)");
        Matcher matcher = pattern.matcher(input);

        // Iterate over matches and extract the captured groups
        while (matcher.find()) {
            String extractedString = matcher.group(1); // Extract the content inside the parentheses
            extractedStrings.add(extractedString);
        }

        return extractedStrings;
    }


    @Test
    public void test_executeScript() {
        String result = ScriptUtils.executeScript(scriptService, "any function", Collections.singletonMap("key", "value"));
        assertEquals("test result", result);
    }
}
