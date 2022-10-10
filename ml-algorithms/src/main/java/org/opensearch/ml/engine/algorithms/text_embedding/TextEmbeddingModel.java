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

    private ModelHelper customModelManager;
    private String modelId;
    public static final String MODEL_ZIP_FILE = "model_zip_file";
    public static final String CUSTOM_MODEL_MANAGER = "custom_model_manager";

    @Override
    public MLOutput predict(MLInputDataset inputDataset, MLModel model) {
        throw new MLException("load model first");
    }

    @Override
    public MLOutput predict(MLInputDataset inputDataset) {
        if (customModelManager == null || modelId == null) {
            throw new MLException("model not loaded");
        }
        return customModelManager.predictTextEmbedding(modelId, inputDataset);
    }

    @Override
    public void initModel(MLModel model, Map<String, Object> params) {
        String engine = model.getModelFormat() == MLModelFormat.TORCH_SCRIPT ? "PyTorch" : "OnnxRuntime";
        File modelZipFile = (File)params.get(MODEL_ZIP_FILE);
        customModelManager = (ModelHelper)params.get(CUSTOM_MODEL_MANAGER);
        modelId = model.getModelId();
        customModelManager.loadModel(
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
        if (customModelManager != null && modelId != null) {
            customModelManager.unloadModel(modelId);
        }
    }
}
