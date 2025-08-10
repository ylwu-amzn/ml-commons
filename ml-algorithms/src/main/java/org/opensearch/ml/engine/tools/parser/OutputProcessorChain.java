/*
* Copyright OpenSearch Contributors
* SPDX-License-Identifier: Apache-2.0
*/

package org.opensearch.ml.engine.tools.parser;

import static org.opensearch.ml.common.utils.StringUtils.gson;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.text.StringEscapeUtils;
import org.opensearch.ml.common.utils.StringUtils;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.reflect.TypeToken;
import com.google.gson.JsonSyntaxException;
import com.jayway.jsonpath.JsonPath;
import com.jayway.jsonpath.PathNotFoundException;

import lombok.extern.log4j.Log4j2;

/**
 * Common framework for processing outputs from ML models and tools
 */
@Log4j2
public class OutputProcessorChain {

    public static final String OUTPUT_PROCESSORS = "output_processors";
    public static final String TO_STRING = "to_string";
    public static final String REGEX_REPLACE = "regex_replace";
    public static final String JSONPATH_FILTER = "jsonpath_filter";
    public static final String EXTRACT_JSON = "extract_json";
    public static final String REGEX_CAPTURE = "regex_capture";

    /**
     * Interface for customized output processors
     */
    public interface OutputProcessor {
        /**
         * Process the input value and return the processed result
         * @param input The object to process
         * @return The processed result
         */
        Object process(Object input);
    }

    /**
     * Registry for creating processor instances from configuration
     */
    public static class ProcessorRegistry {
        private static final Map<String, Function<Map<String, Object>, OutputProcessor>> PROCESSORS = new HashMap<>();

        static {
            // Register all available processors
            registerDefaultProcessors();
        }

        /**
         * Register all built-in processors
         */
        private static void registerDefaultProcessors() {
            // to String
            PROCESSORS.put(TO_STRING, config -> {
                boolean escapeJson = Boolean.TRUE.equals(config.getOrDefault("escape_json", false));

                return inputObj -> {
                    String text = StringUtils.toJson(inputObj);
                    if (escapeJson) {
                        return StringEscapeUtils.escapeJson(text);
                    }
                    return text;
                };
            });

            // Regex replacement processor
            PROCESSORS.put(REGEX_REPLACE, config -> {
                String pattern = (String) config.get("pattern");
                String replacement = (String) config.getOrDefault("replacement", "");
                boolean replaceAll = Boolean.TRUE.equals(config.getOrDefault("replace_all", true));

                return inputObj -> {
                    String text = StringUtils.toJson(inputObj);
                    try {
                        Pattern p = Pattern.compile(pattern, Pattern.DOTALL);
                        if (replaceAll) {
                            return p.matcher(text).replaceAll(replacement);
                        } else {
                            return p.matcher(text).replaceFirst(replacement);
                        }
                    } catch (Exception e) {
                        log.warn("Failed to apply regex: {}", e.getMessage());
                        return inputObj;
                    }
                };
            });

            // JsonPath processor
            PROCESSORS.put(JSONPATH_FILTER, config -> {
                String path = (String) config.get("path");
                Object defaultValue = config.get("default");

                return input -> {
                    try {
                        String jsonStr = StringUtils.toJson(input);
                        return JsonPath.read(jsonStr, path);
                    } catch (PathNotFoundException e) {
                        return defaultValue != null ? defaultValue : input;
                    } catch (Exception e) {
                        log.warn("Failed to apply JsonPath: {}", e.getMessage());
                        return input;
                    }
                };
            });

            // Extract JSON processor
            PROCESSORS.put(EXTRACT_JSON, config -> {
                // Config options
                String extractType = (String) config.getOrDefault("extract_type", "auto"); // "object", "array", or "auto"
                Object defaultValue = config.get("default");

                return input -> {
                    if (!(input instanceof String))
                        return input;
                    String text = (String) input;

                    try {
                        // Find first JSON start char based on config or auto
                        int start = -1;

                        if ("object".equalsIgnoreCase(extractType)) {
                            start = text.indexOf('{');
                        } else if ("array".equalsIgnoreCase(extractType)) {
                            start = text.indexOf('[');
                        } else { // auto detect (default)
                            int startBrace = text.indexOf('{');
                            int startBracket = text.indexOf('[');
                            if (startBrace < 0) {// '{' not found in the string
                                start = startBracket;
                            } else if (startBracket < 0) {// '[' not found in the string
                                start = startBrace;
                            } else {
                                start = Math.min(startBrace, startBracket);
                            }
                        }

                        if (start < 0) {
                            return defaultValue != null ? defaultValue : input;
                        }

                        ObjectMapper mapper = new ObjectMapper();
                        JsonNode jsonNode = mapper.readTree(text.substring(start));

                        if ("object".equalsIgnoreCase(extractType)) {
                            if (jsonNode.isObject()) {
                                return mapper.convertValue(jsonNode, Map.class);
                            } else {
                                return defaultValue != null ? defaultValue : input;
                            }
                        } else if ("array".equalsIgnoreCase(extractType)) {
                            if (jsonNode.isArray()) {
                                return mapper.convertValue(jsonNode, List.class);
                            } else {
                                return defaultValue != null ? defaultValue : input;
                            }
                        } else { // auto
                            if (jsonNode.isObject()) {
                                return mapper.convertValue(jsonNode, Map.class);
                            } else if (jsonNode.isArray()) {
                                return mapper.convertValue(jsonNode, List.class);
                            } else {
                                return defaultValue != null ? defaultValue : input;
                            }
                        }
                    } catch (Exception e) {
                        log.warn("Failed to extract JSON: {}", e.getMessage());
                        return defaultValue != null ? defaultValue : input;
                    }
                };
            });

            // Regex capture processor
            PROCESSORS.put(REGEX_CAPTURE, config -> {
                String pattern = (String) config.get("pattern");
                Object groupsObj = config.getOrDefault("groups", "1");

                // Parse groups into a List<Integer>
                List<Integer> groupIndices = new ArrayList<>();
                try {
                    String groupsStr = groupsObj.toString().trim();
                    boolean isGroups = groupsStr.startsWith("[") && groupsStr.endsWith("]");
                    if (isGroups) {
                        // Multiple group numbers, example: "[1, 2, 4]"
                        String[] parts = groupsStr.substring(1, groupsStr.length() - 1).split(",");
                        for (String part : parts) {
                            groupIndices.add(Integer.parseInt(part.trim()));
                        }
                    } else {
                        // Single group number
                        groupIndices.add(Integer.parseInt(groupsStr));
                    }
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("Invalid 'groups' format: " + groupsObj, e);
                }

                return input -> {
                    String text = StringUtils.toJson(input);

                    try {
                        Pattern p = Pattern.compile(pattern, Pattern.DOTALL);
                        Matcher m = p.matcher(text);
                        if (m.find()) {
                            List<String> captures = new ArrayList<>();
                            for (Integer idx : groupIndices) {
                                if (idx <= m.groupCount()) {
                                    captures.add(m.group(idx));
                                }
                            }
                            if (captures.size() == 1) {
                                return captures.get(0);
                            }
                            return captures;
                            // return String.join(" ", captures); // join results with a space
                        }
                        return input;
                    } catch (Exception e) {
                        log.warn("Failed to apply regex capture: {}", e.getMessage());
                        return input;
                    }
                };
            });

            // Add more processors as needed
        }

        /**
         * Register a custom processor type
         * @param type Processor type identifier
         * @param factory Factory function to create processor instances
         */
        public static void registerProcessor(String type, Function<Map<String, Object>, OutputProcessor> factory) {
            PROCESSORS.put(type, factory);
        }

        /**
         * Create a processor from configuration
         * @param type Processor type
         * @param config Processor configuration
         * @return Configured processor instance
         */
        public static OutputProcessor createProcessor(String type, Map<String, Object> config) {
            Function<Map<String, Object>, OutputProcessor> factory = PROCESSORS.get(type);
            if (factory == null) {
                throw new IllegalArgumentException("Unknown output processor type: " + type);
            }
            return factory.apply(config);
        }

        /**
         * Create a processing chain from a list of processor configurations
         * @param processorConfigs List of processor configurations
         * @return List of configured processors
         */
        @SuppressWarnings("unchecked")
        public static List<OutputProcessor> createProcessingChain(List<Map<String, Object>> processorConfigs) {
            if (processorConfigs == null || processorConfigs.isEmpty()) {
                return Collections.emptyList();
            }

            List<OutputProcessor> processors = new ArrayList<>();
            for (Map<String, Object> config : processorConfigs) {
                String type = (String) config.get("type");
                processors.add(createProcessor(type, config));
            }

            return processors;
        }
    }

    // List of processors to apply sequentially
    private final List<OutputProcessor> processors;

    /**
     * Create a processor chain from configuration
     * @param processorConfigs List of processor configurations
     */
    public OutputProcessorChain(List<Map<String, Object>> processorConfigs) {
        this.processors = ProcessorRegistry.createProcessingChain(processorConfigs);
    }

    /**
     * Create a processor chain from a list of processor instances
     * @param processors List of processor instances
     */
    public OutputProcessorChain(OutputProcessor... processors) {
        this.processors = Arrays.asList(processors);
    }

    /**
     * Process input through the chain of processors
     * @param input Input object to process
     * @return Processed result
     */
    public Object process(Object input) {
        Object result = input;
        for (OutputProcessor processor : processors) {
            result = processor.process(result);
        }
        return result;
    }

    /**
     * Check if this chain has any processors
     * @return true if the chain has at least one processor
     */
    public boolean hasProcessors() {
        return !processors.isEmpty();
    }

    /**
     * Helper method to extract processor configurations from tool parameters
     * @param params Tool parameters
     * @return List of processor configurations or empty list if none found
     */
    @SuppressWarnings("unchecked")
    public static List<Map<String, Object>> extractProcessorConfigs(Map<String, Object> params) {
        if (params == null || !params.containsKey(OUTPUT_PROCESSORS)) {
            return Collections.emptyList();
        }

        Object configObj = params.get(OUTPUT_PROCESSORS);
        if (configObj instanceof List) {
            return (List<Map<String, Object>>) configObj;
        }

        if (configObj instanceof String) {
            String configStr = (String) configObj;
            try {
                List<Map<String, Object>> processorConfigs = gson.fromJson(configStr, new TypeToken<List<Map<String, Object>>>() {
                }.getType());

                if (processorConfigs != null) {
                    return processorConfigs;
                } else {
                    log.warn("Failed to parse output processor config: null result from JSON parsing");
                }
            } catch (JsonSyntaxException e) {
                log.error("Invalid JSON format in output processor configuration: {}", configStr, e);
            } catch (Exception e) {
                log.error("Error parsing output processor configuration: {}", configStr, e);
            }
        }

        return Collections.emptyList();
    }
}
