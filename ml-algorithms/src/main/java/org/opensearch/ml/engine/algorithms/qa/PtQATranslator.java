/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.qa;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

public class PtQATranslator implements Translator<QAInput, String> {
//    private static final int[] AXIS = {0};
    private List<String> tokens;
    private HuggingFaceTokenizer tokenizer;
//    private TextEmbeddingModelConfig.PoolingMode poolingMode;
//    private boolean normalizeResult;
//    private String modelType;

    protected PtQATranslator() {
    }

    @Override
    public Batchifier getBatchifier() {
        return null;
    }

    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        Path path = ctx.getModel().getModelPath();
        tokenizer = HuggingFaceTokenizer.builder().optPadding(true).optTokenizerPath(path.resolve("tokenizer.json")).build();
    }

    @Override
    public NDList processInput(TranslatorContext ctx, QAInput input) {
        NDManager manager = ctx.getNDManager();
        String question = input.getQuestion();
        String paragraph = input.getParagraph();
        NDList ndList = new NDList();

        Encoding encode = tokenizer.encode(question, paragraph);
        tokens = Arrays.asList(encode.getTokens());
        ctx.setAttachment("encoding", encode);
        long[] indices = encode.getIds();
        long[] attentionMask = encode.getAttentionMask();

        NDArray indicesArray = manager.create(indices).expandDims(0);
        indicesArray.setName("input_ids");
        NDArray attentionMaskArray = manager.create(attentionMask).expandDims(0);
        attentionMaskArray.setName("attention_mask");
        ndList.add(indicesArray);
        ndList.add(attentionMaskArray);
//        if ("bert".equalsIgnoreCase(modelType) || "albert".equalsIgnoreCase(modelType)) {
//            long[] tokenTypeIds = encode.getTypeIds();
//            NDArray tokenTypeIdsArray = manager.create(tokenTypeIds).expandDims(0);
//            tokenTypeIdsArray.setName("token_type_ids");
//            ndList.add(tokenTypeIdsArray);
//        }
        return ndList;
    }

    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        NDArray startLogits = list.get(0);
        NDArray endLogits = list.get(1);
        int startIdx = (int) startLogits.argMax().getLong();
        int endIdx = (int) endLogits.argMax().getLong();
        if (startIdx >= endIdx) {
            int tmp = startIdx;
            startIdx = endIdx;
            endIdx = tmp;
        }
        return tokenizer.buildSentence(tokens.subList(startIdx, endIdx + 1));
    }

}
