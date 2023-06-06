/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools;

import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.spi.tools.ToolAnnotation;

import java.util.function.Supplier;

@ToolAnnotation(LanguageModelTool.NAME)
public class LanguageModelTool implements Tool {
    public static final String NAME = "LanguageModelTool";
    private static final String description = "Use this tool for general knowledge question or other general purpose queries and logic. Do NOT use this tool if there are other tools available.";
    private Supplier<ModelTensorOutput> supplier;

    @Override
    public <T> T run(String input) {
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
    public boolean validate(String input) {
        if (input == null || input.length() == 0) {
            return false;
        }
        return true;
    }

    public void setSupplier(Supplier<ModelTensorOutput> supplier) {
        this.supplier = supplier;
    }
}
