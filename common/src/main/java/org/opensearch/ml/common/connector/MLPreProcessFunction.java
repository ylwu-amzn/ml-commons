/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.connector;

import org.opensearch.ml.common.dataset.TextSimilarityInputDataSet;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class MLPreProcessFunction {

    private static final Map<String, Function<?, Map<String, Object>>> PRE_PROCESS_FUNCTIONS = new HashMap<>();
    public static final String TEXT_DOCS_TO_COHERE_EMBEDDING_INPUT = "connector.pre_process.cohere.embedding";
    public static final String TEXT_DOCS_TO_OPENAI_EMBEDDING_INPUT = "connector.pre_process.openai.embedding";

    public static final String TEXT_DOCS_TO_DEFAULT_EMBEDDING_INPUT = "connector.pre_process.default.embedding";
    public static final String TEXT_SIMILARITY_TO_COHERE_RERANK_INPUT = "connector.pre_process.cohere.rerank";
    public static final String TEXT_SIMILARITY_TO_DEFAULT_INPUT = "connector.pre_process.default.rerank";

    private static Function<List<String>, Map<String, Object>> cohereTextEmbeddingPreProcess() {
        return inputs -> Map.of("parameters", Map.of("texts", inputs));
    }

    private static Function<List<String>, Map<String, Object>> openAiTextEmbeddingPreProcess() {
        return inputs -> Map.of("parameters", Map.of("input", inputs));
    }

    private static Function<TextSimilarityInputDataSet, Map<String, Object>> cohereRerankPreProcess() {
        return input -> Map.of("parameters", Map.of(
                "query", input.getQueryText(),
                "documents", input.getTextDocs(),
                "top_n", input.getTextDocs().size()
        ));
    }

    static {
        PRE_PROCESS_FUNCTIONS.put(TEXT_DOCS_TO_COHERE_EMBEDDING_INPUT, cohereTextEmbeddingPreProcess());
        PRE_PROCESS_FUNCTIONS.put(TEXT_DOCS_TO_OPENAI_EMBEDDING_INPUT, openAiTextEmbeddingPreProcess());
        PRE_PROCESS_FUNCTIONS.put(TEXT_DOCS_TO_DEFAULT_EMBEDDING_INPUT, openAiTextEmbeddingPreProcess());
        PRE_PROCESS_FUNCTIONS.put(TEXT_SIMILARITY_TO_DEFAULT_INPUT, cohereRerankPreProcess());
        PRE_PROCESS_FUNCTIONS.put(TEXT_SIMILARITY_TO_COHERE_RERANK_INPUT, cohereRerankPreProcess());
    }

    public static boolean contains(String functionName) {
        return PRE_PROCESS_FUNCTIONS.containsKey(functionName);
    }

    public static <T> Function<T, Map<String, Object>> get(String postProcessFunction) {
        return (Function<T, Map<String, Object>>) PRE_PROCESS_FUNCTIONS.get(postProcessFunction);
    }
}
