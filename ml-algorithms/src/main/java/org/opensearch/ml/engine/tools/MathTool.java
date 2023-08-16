/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools;

import com.google.common.collect.ImmutableMap;
import lombok.Getter;
import lombok.Setter;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.spi.tools.ToolAnnotation;
import org.opensearch.script.ScriptService;

import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.opensearch.ml.engine.utils.ScriptUtils.executeScript;

/**
 * This tool supports running any math expression as painless script.
 */
@ToolAnnotation(MathTool.NAME)
public class MathTool implements Tool {
    public static final String NAME = "MathTool";

    @Setter @Getter
    private String alias;

    @Setter
    private ScriptService scriptService;

    @Getter @Setter
    private String description = "Use this tool to calculate any math problem.";

    public MathTool(ScriptService scriptService) {
        this.scriptService = scriptService;
    }

    @Override
    public <T> T run(Map<String, String> parameters) {
        String input = parameters.get("input");

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
    public boolean validate(Map<String, String> parameters) {
        try {
            run(parameters);
        } catch (Exception e) {
            return false;
        }
        return true;
    }

    public static class Factory implements Tool.Factory<MathTool> {
        private ScriptService scriptService;

        private static Factory INSTANCE;
        public static Factory getInstance() {
            if (INSTANCE != null) {
                return INSTANCE;
            }
            synchronized (MathTool.class) {
                if (INSTANCE != null) {
                    return INSTANCE;
                }
                INSTANCE = new Factory();
                return INSTANCE;
            }
        }

        public void init(ScriptService scriptService) {
            this.scriptService = scriptService;
        }

        @Override
        public MathTool create(Map<String, Object> map) {
            return new MathTool(scriptService);
        }
    }
}