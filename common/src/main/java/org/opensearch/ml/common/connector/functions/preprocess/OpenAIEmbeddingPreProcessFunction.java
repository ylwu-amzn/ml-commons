package org.opensearch.ml.common.connector.functions.preprocess;

import org.opensearch.ml.common.dataset.TextDocsInputDataSet;
import org.opensearch.ml.common.dataset.TextSimilarityInputDataSet;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;

import java.util.Map;

import static org.opensearch.ml.common.utils.StringUtils.convertScriptStringToJsonString;


public class OpenAIEmbeddingPreProcessFunction extends ConnectorPreProcessFunction {

    public OpenAIEmbeddingPreProcessFunction() {
        this.returnDirectlyForRemoteInferenceInput = true;
    }

    @Override
    public void validate(MLInput mlInput) {
        if (!(mlInput.getInputDataset() instanceof TextDocsInputDataSet)) {
            throw new IllegalArgumentException("This pre_process_function can only support TextDocsInputDataSet");
        }
    }

    @Override
    public RemoteInferenceInputDataSet process(MLInput mlInput) {
        TextDocsInputDataSet inputData = (TextDocsInputDataSet) mlInput.getInputDataset();
        Map<String, Object> processedResult = Map.of("parameters", Map.of("input", processTextDocs(inputData)));
        return RemoteInferenceInputDataSet.builder().parameters(convertScriptStringToJsonString(processedResult)).build();
    }
}
