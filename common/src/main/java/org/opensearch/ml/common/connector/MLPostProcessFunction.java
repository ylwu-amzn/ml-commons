/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.connector;

import org.opensearch.ml.common.connector.functions.postprocess.CohereRerankPostProcessFunction;
import org.opensearch.ml.common.connector.functions.postprocess.EmbeddingPostProcessFunction;
import org.opensearch.ml.common.output.model.MLResultDataType;
import org.opensearch.ml.common.output.model.ModelTensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public class MLPostProcessFunction {

    public static final String COHERE_EMBEDDING = "connector.post_process.cohere.embedding";
    public static final String OPENAI_EMBEDDING = "connector.post_process.openai.embedding";
    public static final String COHERE_RERANK = "connector.post_process.cohere.rerank";

    public static final String DEFAULT_EMBEDDING = "connector.post_process.default.embedding";
    public static final String DEFAULT_RERANK = "connector.post_process.default.rerank";

    private static final Map<String, String> JSON_PATH_EXPRESSION = new HashMap<>();

    private static final Map<String, Function<Object, List<ModelTensor>>> POST_PROCESS_FUNCTIONS = new HashMap<>();


    static {
        EmbeddingPostProcessFunction embeddingPostProcessFunction = new EmbeddingPostProcessFunction();
        CohereRerankPostProcessFunction cohereRerankPostProcessFunction = new CohereRerankPostProcessFunction();
        JSON_PATH_EXPRESSION.put(OPENAI_EMBEDDING, "$.data[*].embedding");
        JSON_PATH_EXPRESSION.put(COHERE_EMBEDDING, "$.embeddings");
        JSON_PATH_EXPRESSION.put(DEFAULT_EMBEDDING, "$[*]");
        JSON_PATH_EXPRESSION.put(COHERE_RERANK, "$.results");
        JSON_PATH_EXPRESSION.put(DEFAULT_RERANK, "$[*]");
        POST_PROCESS_FUNCTIONS.put(OPENAI_EMBEDDING, embeddingPostProcessFunction);
        POST_PROCESS_FUNCTIONS.put(COHERE_EMBEDDING, embeddingPostProcessFunction);
        POST_PROCESS_FUNCTIONS.put(DEFAULT_EMBEDDING, embeddingPostProcessFunction);
        POST_PROCESS_FUNCTIONS.put(COHERE_RERANK, cohereRerankPostProcessFunction);
        POST_PROCESS_FUNCTIONS.put(DEFAULT_RERANK, cohereRerankPostProcessFunction);
    }

//    public static Function<Object, List<ModelTensor>> buildEmbeddingModelTensorList() {
//        return input -> {
//            if (input == null) {
//                throw new IllegalArgumentException("The list of embeddings is null when using the built-in post-processing function.");
//            }
//            List<List<Float>> embeddings = (List<List<Float>>) input;
//            List<ModelTensor> modelTensors = new ArrayList<>();
//            embeddings.forEach(embedding -> modelTensors.add(
//                ModelTensor
//                    .builder()
//                    .name("sentence_embedding")
//                    .dataType(MLResultDataType.FLOAT32)
//                    .shape(new long[]{embedding.size()})
//                    .data(embedding.toArray(new Number[0]))
//                    .build()
//            ));
//            return modelTensors;
//        };
//    }
//
//    public static Function<Object, List<ModelTensor>> buildCohereRerankModelTensorList() {
//        return input -> {
//            if (input == null) {
//                throw new IllegalArgumentException("The Cohere rerank result is null when using the built-in post-processing function.");
//            }
//            List<ModelTensor> modelTensors = new ArrayList<>();
//
//            List<Map<String,Object>> rerankResults = ((List<Map<String,Object>>)input);
//
//            if (rerankResults.size() > 0) {
//                Double[] scores = new Double[rerankResults.size()];
//                for (int i = 0; i < rerankResults.size(); i++) {
//                    Integer index = (Integer) rerankResults.get(i).get("index");
//                    scores[index] = (Double) rerankResults.get(i).get("relevance_score");
//                }
//
//                for (int i = 0; i < scores.length; i++) {
//                    modelTensors.add(ModelTensor.builder()
//                            .name("similarity")
//                            .shape(new long[]{1})
//                            .data(new Number[]{scores[i]})
//                            .dataType(MLResultDataType.FLOAT32)
//                            .build());
//                }
//            }
//
//
//            return modelTensors;
//        };
//    }

    public static String getResponseFilter(String postProcessFunction) {
        return JSON_PATH_EXPRESSION.get(postProcessFunction);
    }

    public static Function<Object, List<ModelTensor>> get(String postProcessFunction) {
        return POST_PROCESS_FUNCTIONS.get(postProcessFunction);
    }

    public static boolean contains(String postProcessFunction) {
        return POST_PROCESS_FUNCTIONS.containsKey(postProcessFunction);
    }
}
