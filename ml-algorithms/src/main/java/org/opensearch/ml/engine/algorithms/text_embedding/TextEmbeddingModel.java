package org.opensearch.ml.engine.algorithms.text_embedding;

import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.dataset.MLInputDataset;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.model.MLModelFormat;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.engine.ModelHelper;
import org.opensearch.ml.engine.Predictable;
import org.opensearch.ml.engine.annotation.Function;

import java.io.File;
import java.util.Map;

@Function(FunctionName.TEXT_EMBEDDING)
public class TextEmbeddingModel implements Predictable {

    private ModelHelper modelHelper;
    private String modelId;
    public static final String MODEL_ZIP_FILE = "model_zip_file";
    public static final String MODEL_HELPER = "model_helper";

    @Override
    public MLOutput predict(MLInputDataset inputDataset, MLModel model) {
        throw new MLException("load model first");
    }

    @Override
    public MLOutput predict(MLInputDataset inputDataset) {
        if (modelHelper == null || modelId == null) {
            throw new MLException("model not loaded");
        }
        return modelHelper.predictTextEmbedding(modelId, inputDataset);
    }

    @Override
    public void initModel(MLModel model, Map<String, Object> params) {
        String engine = model.getModelFormat() == MLModelFormat.TORCH_SCRIPT ? "PyTorch" : "OnnxRuntime";
        File modelZipFile = (File)params.get(MODEL_ZIP_FILE);
        modelHelper = (ModelHelper)params.get(MODEL_HELPER);
        modelId = model.getModelId();
        modelHelper.loadModel(
                modelZipFile,
                modelId,
                model.getName(),
                model.getModelTaskType(),
                model.getVersion(),
                model.getModelConfig(),
                engine
        );
    }

    @Override
    public void close() {
        if (modelHelper != null && modelId != null) {
            modelHelper.unloadModel(modelId);
        }
    }
}
