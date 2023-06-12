/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools;

import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.spi.tools.ToolAnnotation;

import java.util.Map;
import java.util.function.Supplier;

@ToolAnnotation(LanguageModelTool.NAME)
public class LanguageModelTool implements Tool {
    public static final String NAME = "LanguageModelTool";
    private static final String description = "Useful for answering any general questions.";
    private Supplier<ModelTensorOutput> supplier;

    @Override
    public <T> T run(String input, Map<String, String> toolParameters) {
        ModelTensorOutput output = supplier.get();
        String result = (String) output.getMlModelOutputs().get(0).getMlModelTensors().get(0).getDataAsMap().get("response");
        return (T)result;
    }

    @Override
    public String getName() {
        return LanguageModelTool.NAME;
    }

    @Override
    public String getDescription() {
        return description;
    }

    @Override
    public boolean validate(String input, Map<String, String> toolParameters) {
        if (input == null || input.length() == 0) {
            return false;
        }
        return true;
    }

    public void setSupplier(Supplier<ModelTensorOutput> supplier) {
        this.supplier = supplier;
    }
}
