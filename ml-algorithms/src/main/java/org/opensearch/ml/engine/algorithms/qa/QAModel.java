/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.qa;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import lombok.extern.log4j.Log4j2;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.dataset.MLInputDataset;
import org.opensearch.ml.common.dataset.TextDocsInputDataSet;
import org.opensearch.ml.common.model.MLModelConfig;
import org.opensearch.ml.common.output.model.ModelResultFilter;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.engine.algorithms.DLModel;
import org.opensearch.ml.engine.annotation.Function;

import java.util.ArrayList;
import java.util.List;

@Log4j2
@Function(FunctionName.QUESTION_ANSWERING)
public class QAModel extends DLModel {

    @Override
    public ModelTensorOutput predict(String modelId, MLInputDataset inputDataSet) throws TranslateException {
        List<ModelTensors> tensorOutputs = new ArrayList<>();
        Output output;
        TextDocsInputDataSet textDocsInput = (TextDocsInputDataSet) inputDataSet;
        ModelResultFilter resultFilter = textDocsInput.getResultFilter();
        int currentDevice = getCurrentDevice();
        List<String> docs = textDocsInput.getDocs();
        if (docs == null || docs.size() < 2) {
            throw new IllegalArgumentException("wrong doc count");
        }
        //TODO: support multiple question/paragraph pairs
        String question = docs.get(0);
        String paragraph = docs.get(1);
        Input input = new Input();
        input.add("question", question);
        input.add("paragraph", paragraph);
        output = predictors[currentDevice].predict(input);
        tensorOutputs.add(parseModelTensorOutput(output, resultFilter));
        return new ModelTensorOutput(tensorOutputs);
    }

    @Override
    public Translator<Input, Output> getTranslator(String engine, MLModelConfig modelConfig) {
        return new PtQATranslator();
    }

}
