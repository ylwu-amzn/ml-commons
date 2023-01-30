package org.opensearch.ml.engine.algorithms.time_series;

import ai.djl.Model;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import lombok.extern.log4j.Log4j2;

import java.lang.reflect.Type;
import java.util.Collections;
import java.util.Map;
import java.util.Set;

@Log4j2
public class MyDeepARTranslatorFactory implements TranslatorFactory {

    @Override
    public Set<Pair<Type, Type>> getSupportedTypes() {
        return Collections.singleton(new Pair<>(Input.class, Output.class));
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public <I, O> Translator<I, O> newInstance(
            Class<I> input, Class<O> output, Model model, Map<String, ?> arguments) {
        if (!isSupported(input, output)) {
            throw new IllegalArgumentException("Unsupported input/output types.");
        }
        return (Translator<I, O>) DeepARTranslator.builder(arguments).build();
    }
}
