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
    public static final String TEXT_DOCS_TO_COHERE_EMBEDDING_INPUT = "connector.pre_process.cohere.embedding";
    public static final String TEXT_DOCS_TO_OPENAI_EMBEDDING_INPUT = "connector.pre_process.openai.embedding";

    public static final String TEXT_DOCS_TO_DEFAULT_EMBEDDING_INPUT = "connector.pre_process.default.embedding";
    public static final String TEXT_SIMILARITY_TO_COHERE_RERANK_INPUT = "connector.pre_process.cohere.rerank";
    public static final String TEXT_SIMILARITY_TO_DEFAULT_INPUT = "connector.pre_process.default.rerank";

    public static final String PROCESS_REMOTE_INFERENCE_INPUT = "pre_process_function.process_remote_inference_input";

    static {
        CohereEmbeddingPreProcessFunction cohereEmbeddingPreProcessFunction = new CohereEmbeddingPreProcessFunction();
        OpenAIEmbeddingPreProcessFunction openAIEmbeddingPreProcessFunction = new OpenAIEmbeddingPreProcessFunction();
        CohereRerankPreProcessFunction cohereRerankPreProcessFunction = new CohereRerankPreProcessFunction();
        PRE_PROCESS_FUNCTIONS.put(TEXT_DOCS_TO_COHERE_EMBEDDING_INPUT, cohereEmbeddingPreProcessFunction);
        PRE_PROCESS_FUNCTIONS.put(TEXT_DOCS_TO_OPENAI_EMBEDDING_INPUT, openAIEmbeddingPreProcessFunction);
        PRE_PROCESS_FUNCTIONS.put(TEXT_DOCS_TO_DEFAULT_EMBEDDING_INPUT, openAIEmbeddingPreProcessFunction);
        PRE_PROCESS_FUNCTIONS.put(TEXT_SIMILARITY_TO_DEFAULT_INPUT, cohereRerankPreProcessFunction);
        PRE_PROCESS_FUNCTIONS.put(TEXT_SIMILARITY_TO_COHERE_RERANK_INPUT, cohereRerankPreProcessFunction);
    }

    public static boolean contains(String functionName) {
        return PRE_PROCESS_FUNCTIONS.containsKey(functionName);
    }

    public static Function<MLInput, RemoteInferenceInputDataSet> get(String postProcessFunction) {
        return PRE_PROCESS_FUNCTIONS.get(postProcessFunction);
    }
}
