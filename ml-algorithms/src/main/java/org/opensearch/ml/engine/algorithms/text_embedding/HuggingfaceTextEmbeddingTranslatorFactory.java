/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.text_embedding;

import ai.djl.Model;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;

import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class HuggingfaceTextEmbeddingTranslatorFactory implements TranslatorFactory {

    private static final Set<Pair<Type, Type>> SUPPORTED_TYPES = new HashSet<>();

    static {
        SUPPORTED_TYPES.add(new Pair<>(String.class, float[].class));
        SUPPORTED_TYPES.add(new Pair<>(Input.class, Output.class));
    }

    private final String poolingMode;
    private boolean normalizeResult;
    private final String modelType;
    private final boolean neuron;

    public HuggingfaceTextEmbeddingTranslatorFactory(String poolingMode, boolean normalizeResult, String modelType, boolean neuron) {
        this.poolingMode = poolingMode;
        this.normalizeResult = normalizeResult;
        this.modelType = modelType;
        this.neuron = neuron;
    }

    /** {@inheritDoc} */
    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return SUPPORTED_TYPES;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Model model, Map<String, ?> arguments)
            throws TranslateException {
        Path modelPath = model.getModelPath();
        try {
            HuggingFaceTokenizer tokenizer =
                    HuggingFaceTokenizer.builder(arguments)
                            .optTokenizerPath(modelPath)
                            .optManager(model.getNDManager())
                            .build();
            boolean withTokenTypeIdsInput = false;
            if (neuron && ("bert".equalsIgnoreCase(modelType) || "albert".equalsIgnoreCase(modelType))) {
                withTokenTypeIdsInput = true;
            }
            HuggingfaceTextEmbeddingTranslator translator =
                    HuggingfaceTextEmbeddingTranslator.builder(tokenizer, arguments)
                            //.optPoolingMethod(poolingMode)
                            .optPoolingMode(poolingMode)
                            .optNormalize(normalizeResult)
                            //.optWithTokenTypeIdsInput(withTokenTypeIdsInput)
                            .build();
            if (input == String.class && output == float[].class) {
                return (Translator<I, O>) translator;
            } else if (input == Input.class && output == Output.class) {
                return (Translator<I, O>) new HuggingfaceTextEmbeddingServingTranslator(translator);
            }
            throw new IllegalArgumentException("Unsupported input/output types.");
        } catch (IOException e) {
            throw new TranslateException("Failed to load tokenizer.", e);
        }
    }
}
