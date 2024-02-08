/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.utils;

import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.gson.Gson;
import com.jayway.jsonpath.JsonPath;
import org.opensearch.ml.common.utils.StringUtils;
import org.opensearch.script.Script;
import org.opensearch.script.ScriptService;
import org.opensearch.script.ScriptType;
import org.opensearch.script.TemplateScript;

import com.google.common.collect.ImmutableMap;

import static org.opensearch.ml.common.utils.StringUtils.gson;

public class ScriptUtils {

    public static Optional<String> executePreprocessFunction(
        ScriptService scriptService,
        String preProcessFunction,
        List<String> inputSentences
    ) {
        return Optional.ofNullable(executeScript(scriptService, preProcessFunction, ImmutableMap.of("text_docs", inputSentences)));
    }

    public static Optional<String> executePostProcessFunction(ScriptService scriptService, String postProcessFunction, String resultJson) {
        Map<String, Object> result = StringUtils.fromJson(resultJson, "result");
//        for (String key : result.keySet()) {
//            Object o = result.get(key);
//            if () {
//            }
//        }
//        String p = null;
        /*String newPostProcessFunction = postProcessFunction;
        try {
            List<String> filters = extractJsonPathFilter(newPostProcessFunction);
            for (String filter : filters) {
                Object filteredOutput = JsonPath.read(resultJson, filter);
                String filteredResult = escape(filteredOutput);
                newPostProcessFunction = newPostProcessFunction.replace("process_function.json_path_filter(" + filter + ")", filteredResult);
            }
            newPostProcessFunction = newPostProcessFunction.replace("\\n","\\\\n");
            newPostProcessFunction = newPostProcessFunction.replace("\\\"","\\\\\\\"");
        } catch (PrivilegedActionException e) {
            throw new RuntimeException(e);
        }
//        result.put("text", p);
        if (newPostProcessFunction != null) {
            return Optional.ofNullable(executeScript(scriptService, newPostProcessFunction, result));
        }*/
        if (postProcessFunction != null) {
            return Optional.ofNullable(executeScript(scriptService, postProcessFunction, result));
        }
        return Optional.empty();
    }

    public static String escape(Object obj) throws PrivilegedActionException {
        String json = AccessController.doPrivileged((PrivilegedExceptionAction<String>) () -> gson.toJson(obj));
        if (obj instanceof String) {
            return json;
        }else {
            return AccessController.doPrivileged((PrivilegedExceptionAction<String>) () -> gson.toJson(json));
        }
    }



    static List<String> extractJsonPathFilter(String input) {
        List<String> filters = new ArrayList<>();

        // Define the pattern to match strings following the specified pattern
        Pattern pattern = Pattern.compile("process_function\\.json_path_filter\\((.*?)\\)");
        Matcher matcher = pattern.matcher(input);

        // Iterate over matches and extract the captured groups
        while (matcher.find()) {
            String extractedString = matcher.group(1); // Extract the content inside the parentheses
            filters.add(extractedString);
        }

        return filters;
    }

    public static String executeScript(ScriptService scriptService, String painlessScript, Map<String, Object> params) {
        Script script = new Script(ScriptType.INLINE, "painless", painlessScript, Collections.emptyMap());
        TemplateScript templateScript = scriptService.compile(script, TemplateScript.CONTEXT).newInstance(params);
        return templateScript.execute();
    }
}
