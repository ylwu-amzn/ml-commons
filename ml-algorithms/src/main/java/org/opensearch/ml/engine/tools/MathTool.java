/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools;

import com.google.common.collect.ImmutableMap;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.spi.tools.ToolAnnotation;
import org.opensearch.script.ScriptService;

import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.opensearch.ml.engine.utils.ScriptUtils.executeScript;
import static org.opensearch.ml.engine.utils.ScriptUtils.gson;

@ToolAnnotation(MathTool.NAME)
public class MathTool implements Tool {
    public static final String NAME = "MathTool";
    private ScriptService scriptService;
    private static final String description = "Use this tool to calculate any math problem.";

    public MathTool(ScriptService scriptService) {
        this.scriptService = scriptService;
    }

    @Override
    public <T> T run(String input, Map<String, String> toolParameters) {
        try {
            input = gson.fromJson(input, String.class);
        } catch (Exception e) {
            //e.printStackTrace();
        }
        input = input.replaceAll(",", "");
        if (input.contains("/")) {
            String patternStr = "\\d+(\\.\\d+)?";
            Pattern pattern = Pattern.compile(patternStr);
            Matcher matcher = pattern.matcher(input);
            if (matcher.find()) {
                String match = matcher.group(0);
                double result = Double.parseDouble(match);
                input = input.replaceFirst(patternStr, result+"");
            }
        }
        String result = executeScript(scriptService, input + "+ \"\"", ImmutableMap.of());
        return (T) ("Answer: " + result);
    }

    @Override
    public String getName() {
        return MathTool.NAME;
    }

    @Override
    public String getDescription() {
        return description;
    }

    @Override
    public boolean validate(String input, Map<String, String> toolParameters) {
        try {
            run(input, toolParameters);
        } catch (Exception e) {
            return false;
        }
        return true;
    }
}
