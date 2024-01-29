/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.connector;

import org.opensearch.ml.common.connector.functions.preprocess.CohereEmbeddingPreProcessFunction;
import org.opensearch.ml.common.connector.functions.preprocess.CohereRerankPreProcessFunction;
import org.opensearch.ml.common.connector.functions.preprocess.OpenAIEmbeddingPreProcessFunction;
import org.opensearch.ml.common.dataset.TextDocsInputDataSet;
import org.opensearch.ml.common.dataset.TextSimilarityInputDataSet;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.opensearch.ml.common.utils.StringUtils.gson;

public class MLPreProcessFunction {

    private static final Map<String, Function<MLInput, RemoteInferenceInputDataSet>> PRE_PROCESS_FUNCTIONS = new HashMap<>();
//    private static final Map<String, Function<?, Map<String, Object>>> PRE_PROCESS_FUNCTIONS = new HashMap<>();
    public static final String TEXT_DOCS_TO_COHERE_EMBEDDING_INPUT = "connector.pre_process.cohere.embedding";
    public static final String TEXT_DOCS_TO_OPENAI_EMBEDDING_INPUT = "connector.pre_process.openai.embedding";

    public static final String TEXT_DOCS_TO_DEFAULT_EMBEDDING_INPUT = "connector.pre_process.default.embedding";
    public static final String TEXT_SIMILARITY_TO_COHERE_RERANK_INPUT = "connector.pre_process.cohere.rerank";
    public static final String TEXT_SIMILARITY_TO_DEFAULT_INPUT = "connector.pre_process.default.rerank";

//    private static Function<MLInput, Map<String, Object>> cohereTextEmbeddingPreProcess() {
//        return input -> {
//            if (!(input.getInputDataset() instanceof TextDocsInputDataSet)) {
//                throw new IllegalArgumentException("This pre_process_function can only support TextDocsInputDataSet");
//            }
//            return Map.of("parameters", Map.of("texts", processTextDocs((TextDocsInputDataSet)input.getInputDataset())));
//        };
//    }
//
//    private static Function<MLInput, Map<String, Object>> openAiTextEmbeddingPreProcess() {
//        return input -> {
//            if (!(input.getInputDataset() instanceof TextDocsInputDataSet)) {
//                throw new IllegalArgumentException("This pre_process_function can only support TextDocsInputDataSet");
//            }
//            return Map.of("parameters", Map.of("input", processTextDocs((TextDocsInputDataSet)input.getInputDataset())));
//        };
//    }
//
//    private static Function<MLInput, Map<String, Object>> cohereRerankPreProcess() {
//        return input -> {
//            if (input.getInputDataset() instanceof RemoteInferenceInputDataSet) {
//                return ((RemoteInferenceInputDataSet)input.getInputDataset()).getParameters();
//            }
//            if (!(input.getInputDataset() instanceof TextSimilarityInputDataSet)) {
//                throw new IllegalArgumentException("This pre_process_function can only support TextSimilarityInputDataSet");
//            }
//            TextSimilarityInputDataSet inputData = (TextSimilarityInputDataSet)input.getInputDataset();
//            return Map.of("parameters", Map.of(
//                    "query", inputData.getQueryText(),
//                    "documents", inputData.getTextDocs(),
//                    "top_n", inputData.getTextDocs().size()
//            ));
//        };
//    }
//
//    private static List<String> processTextDocs(TextDocsInputDataSet inputDataSet) {
//        List<String> docs = new ArrayList<>();
//        for (String doc : inputDataSet.getDocs()) {
//            if (doc != null) {
//                String gsonString = gson.toJson(doc);
//                // in 2.9, user will add " before and after string
//                // gson.toString(string) will add extra " before after string, so need to remove
//                docs.add(gsonString.substring(1, gsonString.length() - 1));
//            } else {
//                docs.add(null);
//            }
//        }
//        return docs;
//    }

    static {
        PRE_PROCESS_FUNCTIONS.put(TEXT_DOCS_TO_COHERE_EMBEDDING_INPUT, new CohereEmbeddingPreProcessFunction());
        PRE_PROCESS_FUNCTIONS.put(TEXT_DOCS_TO_OPENAI_EMBEDDING_INPUT, new OpenAIEmbeddingPreProcessFunction());
        PRE_PROCESS_FUNCTIONS.put(TEXT_DOCS_TO_DEFAULT_EMBEDDING_INPUT, new OpenAIEmbeddingPreProcessFunction());
        PRE_PROCESS_FUNCTIONS.put(TEXT_SIMILARITY_TO_DEFAULT_INPUT, new CohereRerankPreProcessFunction());
        PRE_PROCESS_FUNCTIONS.put(TEXT_SIMILARITY_TO_COHERE_RERANK_INPUT, new CohereRerankPreProcessFunction());
    }

//    private static Function<List<String>, Map<String, Object>> cohereTextEmbeddingPreProcess() {
//        return inputs -> Map.of("parameters", Map.of("texts", inputs));
//    }
//
//    private static Function<List<String>, Map<String, Object>> openAiTextEmbeddingPreProcess() {
//        return inputs -> Map.of("parameters", Map.of("input", inputs));
//    }
//
//    private static Function<TextSimilarityInputDataSet, Map<String, Object>> cohereRerankPreProcess() {
//        return input -> Map.of("parameters", Map.of(
//                "query", input.getQueryText(),
//                "documents", input.getTextDocs(),
//                "top_n", input.getTextDocs().size()
//        ));
//    }

//    static {
//        PRE_PROCESS_FUNCTIONS.put(TEXT_DOCS_TO_COHERE_EMBEDDING_INPUT, cohereTextEmbeddingPreProcess());
//        PRE_PROCESS_FUNCTIONS.put(TEXT_DOCS_TO_OPENAI_EMBEDDING_INPUT, openAiTextEmbeddingPreProcess());
//        PRE_PROCESS_FUNCTIONS.put(TEXT_DOCS_TO_DEFAULT_EMBEDDING_INPUT, openAiTextEmbeddingPreProcess());
//        PRE_PROCESS_FUNCTIONS.put(TEXT_SIMILARITY_TO_DEFAULT_INPUT, cohereRerankPreProcess());
//        PRE_PROCESS_FUNCTIONS.put(TEXT_SIMILARITY_TO_COHERE_RERANK_INPUT, cohereRerankPreProcess());
//    }

    public static boolean contains(String functionName) {
        return PRE_PROCESS_FUNCTIONS.containsKey(functionName);
    }

//    public static <T> Function<T, Map<String, Object>> get(String postProcessFunction) {
//        return (Function<T, Map<String, Object>>) PRE_PROCESS_FUNCTIONS.get(postProcessFunction);
//    }

    public static Function<MLInput, RemoteInferenceInputDataSet> get(String postProcessFunction) {
        return PRE_PROCESS_FUNCTIONS.get(postProcessFunction);
    }
}
