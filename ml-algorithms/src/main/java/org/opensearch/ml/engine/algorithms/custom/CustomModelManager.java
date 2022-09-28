/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.custom;

import ai.djl.Application;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.DownloadUtils;
import ai.djl.training.util.ProgressBar;
import ai.djl.util.JsonUtils;
import ai.djl.util.ZipUtils;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hashing;
import com.google.common.io.ByteSource;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FileUtils;
import org.opensearch.action.ActionListener;
import org.opensearch.common.bytes.BytesReference;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.ml.common.dataset.TextDocsInputDataSet;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.exception.MLResourceNotFoundException;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.model.MLModelConfig;
import org.opensearch.ml.common.model.MLModelTaskType;
import org.opensearch.ml.common.model.MLResultDataType;
import org.opensearch.ml.common.model.TextEmbeddingModelConfig;
import org.opensearch.ml.common.output.model.ModelResultFilter;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.transport.custom_model.unload.UnloadModelInput;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static org.opensearch.ml.common.model.TextEmbeddingModelConfig.FrameworkType.SENTENCE_TRANSFORMERS;
import static org.opensearch.ml.engine.MLEngine.DJL_CACHE_PATH;
import static org.opensearch.ml.engine.MLEngine.getCustomModelPath;
import static org.opensearch.ml.engine.MLEngine.getLoadModelPath;
import static org.opensearch.ml.engine.MLEngine.getUploadModelPath;
import static org.opensearch.ml.engine.algorithms.custom.MLTextEmbeddingServingTranslator.toFloats;
import static org.opensearch.ml.engine.algorithms.custom.SentenceTransformerTextEmbeddingTranslator.TEXT_FIELDS;
import static org.opensearch.ml.engine.utils.MLFileUtils.deleteFileQuietly;
import static org.opensearch.ml.engine.utils.MLFileUtils.readAndFragment;

@Log4j2
public class CustomModelManager {
    public static final String CHUNK_FILES = "chunk_files";
    public static final String MODEL_SIZE_IN_BYTES = "model_size_in_bytes";
    public static final String MODEL_FILE_HASH = "model_file_hash";
    public static final String EMBEDDING = "sentence_embedding";
    private static final int CHUNK_SIZE = 10_000_000; // 10MB

    private Map<String, Predictor> predictors;
    private Map<String, TextEmbeddingModelConfig.FrameworkType> modelTransformersTypes;
    private Map<String, ZooModel> models;

    public CustomModelManager() {
        modelTransformersTypes = new ConcurrentHashMap<>();
        predictors = new ConcurrentHashMap<>();
        models = new ConcurrentHashMap<>();
    }

    public void downloadAndSplit(String modelId, String modelName, Integer version, String url, ActionListener<Map<String, Object>> listener) throws PrivilegedActionException {
        try {
            AccessController.doPrivileged((PrivilegedExceptionAction<Void>) () -> {
                Path modelUploadPath = getUploadModelPath(modelId, modelName, version);
                String modelPath = modelUploadPath +".zip";
                Path modelPartsPath = modelUploadPath.resolve("chunks");
                File modelZipFile = new File(modelPath);
                DownloadUtils.download(url, modelPath, new ProgressBar());
                ArrayList<String> chunkFiles = readAndFragment(modelZipFile, modelPartsPath, CHUNK_SIZE);
                Map<String, Object> result = new HashMap<>();
                result.put(CHUNK_FILES, chunkFiles);
                result.put(MODEL_SIZE_IN_BYTES, modelZipFile.length());

                ByteSource byteSource = com.google.common.io.Files.asByteSource(modelZipFile);
                HashCode hc = byteSource.hash(Hashing.md5());
                result.put(MODEL_FILE_HASH, hc.toString());
                FileUtils.delete(modelZipFile);
                listener.onResponse(result);
                return null;
            });
        } catch (Exception e) {
            listener.onFailure(e);
        }
    }

    public void loadModel(File modelZipFile, String modelId, String modelName, MLModelTaskType modelTaskType, Integer version,
                          MLModelConfig modelConfig,
                          String engine) throws PrivilegedActionException {
        AccessController.doPrivileged((PrivilegedExceptionAction<Void>) () -> {
            ClassLoader contextClassLoader = Thread.currentThread().getContextClassLoader();
            try {
                System.setProperty("PYTORCH_PRECXX11", "true");
                System.setProperty("DJL_CACHE_DIR", DJL_CACHE_PATH.toAbsolutePath().toString());
                System.setProperty("java.library.path", DJL_CACHE_PATH.toAbsolutePath().toString());
                Thread.currentThread().setContextClassLoader(ai.djl.Model.class.getClassLoader());
                Engine.debugEnvironment();
                Path modelPath = getCustomModelPath(modelId, modelName, version);
                File pathFile = new File(modelPath.toUri());
                if (pathFile.exists()) {
                    FileUtils.deleteDirectory(pathFile);
                }
                ZipUtils.unzip(new FileInputStream(modelZipFile), modelPath);
                boolean findModelFile = false;
                for (File file : pathFile.listFiles()) {
                    String name = file.getName();
                    if (name.endsWith(".pt") || name.endsWith(".onnx")) {
                        if (findModelFile) {
                            throw new IllegalArgumentException("found multiple models");
                        }
                        findModelFile = true;
                        String suffix = name.substring(name.lastIndexOf("."));
                        file.renameTo(new File(modelPath.resolve(modelName + suffix).toUri()));
                    }
                }
                try {
                    Map<String, Object> arguments = new HashMap<>();
                    arguments.put("engine", engine);
                    Criteria.Builder<Input, Output> criteriaBuilder = Criteria.builder()
                            .setTypes(Input.class, Output.class)
                            .optApplication(Application.UNDEFINED)
                            .optArguments(arguments)
                            .optModelPath(modelPath);
                    if (modelTaskType == MLModelTaskType.TEXT_EMBEDDING) {
                        TextEmbeddingModelConfig textEmbeddingModelConfig = (TextEmbeddingModelConfig) modelConfig;
                        TextEmbeddingModelConfig.FrameworkType transformersType = textEmbeddingModelConfig.getFrameworkType();
                        if (transformersType == SENTENCE_TRANSFORMERS) {
                            criteriaBuilder.optTranslator(new SentenceTransformerTextEmbeddingTranslator());
                        } else {
                            criteriaBuilder.optTranslatorFactory(new MLTextEmbeddingTranslatorFactory());
                        }
                        modelTransformersTypes.put(modelId, transformersType);
                    }
                    Criteria<Input, Output> criteria = criteriaBuilder.build();
                    ZooModel<Input, Output> model = criteria.loadModel();
                    Predictor<Input, Output> predictor = model.newPredictor();
                    predictors.put(modelId, predictor);
                    models.put(modelId, model);
                } catch (Exception e) {
                    String errorMessage = "Failed to load model " + modelName + ", version: " + version;
                    log.error(errorMessage, e);
                    removeModel(modelId);
                    throw new MLException(errorMessage);
                } finally {
                    deleteFileQuietly(getLoadModelPath(modelId));
                }
                return null;
            } finally {
                Thread.currentThread().setContextClassLoader(contextClassLoader);
            }
        });
    }

    private void removeModel(String modelId) {
        predictors.remove(modelId);
        models.remove(modelId);
        modelTransformersTypes.remove(modelId);
    }

    public ModelTensorOutput predict(String modelId, MLInput mlInput) throws IOException, PrivilegedActionException {
        return AccessController.doPrivileged((PrivilegedExceptionAction<ModelTensorOutput>) () -> {
            Thread.currentThread().setContextClassLoader(getClass().getClassLoader());
            if (!predictors.containsKey(modelId)) {
                throw new MLResourceNotFoundException("Model not loaded.");
            }
            Predictor<Input, Output> predictor = predictors.get(modelId);
            MLModelTaskType mlModelTaskType = mlInput.getMlModelTaskType();
            List<ModelTensors> tensorOutputs = new ArrayList<>();
            Output output;
            if (mlModelTaskType != null) {
                switch (mlModelTaskType) {
                    case TEXT_EMBEDDING:
                        TextDocsInputDataSet textDocsInput = (TextDocsInputDataSet) mlInput.getInputDataset();
                        ModelResultFilter resultFilter = ((TextDocsInputDataSet) mlInput.getInputDataset()).getResultFilter();
                        if (modelTransformersTypes.get(modelId) == SENTENCE_TRANSFORMERS) {
                            Input input = new Input();
                            input.add(TEXT_FIELDS, JsonUtils.GSON.toJson(textDocsInput.getDocs()));
                            output = predictor.predict(input);
                            parseModelTensorOutput(output, resultFilter, tensorOutputs);
                        } else {
                            for (String doc : textDocsInput.getDocs()) {
                                Input input = new Input();
                                input.add(doc);
                                output = predictor.predict(input);
                                byte[] bytes = output.getData().getAsBytes();
                                Number[] data = toFloats(bytes);
                                List<ModelTensor> modelTensors = new ArrayList<>();
                                ByteBuffer byteBuffer = null;
                                if (resultFilter.isReturnBytes()) {
                                    byteBuffer = ByteBuffer.wrap(bytes);
                                    byteBuffer.order(ByteOrder.nativeOrder());
                                }
                                ModelTensor modelTensor = new ModelTensor(EMBEDDING, data, new long[]{1, data.length}, MLResultDataType.FLOAT32, byteBuffer);
                                modelTensors.add(modelTensor);
                                ModelTensors mlModelTensorOutput = new ModelTensors(modelTensors);
                                mlModelTensorOutput.filter(resultFilter);
                                tensorOutputs.add(mlModelTensorOutput);
                            }
                        }
                        break;
                    default:
                        throw new IllegalArgumentException("unknown model task type");
                }
            }
            return new ModelTensorOutput(tensorOutputs);
        });
    }

    public Map<String, String> unloadModel(UnloadModelInput unloadModelInput) {
        Map<String, String> modelUnloadStatus = new HashMap<>();
        String[] modelIds = unloadModelInput.getModelIds();
        if (modelIds != null && modelIds.length > 0) {
            for (String modelId : modelIds) {
                deleteFileQuietly(getCustomModelPath(modelId));
                deleteFileQuietly(getLoadModelPath(modelId));
                deleteFileQuietly(getUploadModelPath(modelId));
                if (predictors.containsKey(modelId)) {
                    predictors.get(modelId).close();
                    predictors.remove(modelId);
                    modelUnloadStatus.put(modelId, "deleted");
                } else {
                    modelUnloadStatus.put(modelId, "not_found");
                }
                if (models.containsKey(modelId)) {
                    models.get(modelId).close();
                    models.remove(modelId);
                }
                log.info("Unload model {}", modelId);
            }
        }
        return modelUnloadStatus;
    }

    private void parseModelTensorOutput(Output output, ModelResultFilter resultFilter, List<ModelTensors> tensorOutputs) {
        if (output == null) {
            throw new MLException("No output generated");
        }
        byte[] bytes = output.getData().getAsBytes();
        ByteBuffer byteBuffer = ByteBuffer.wrap(bytes);
        try (StreamInput streamInput = BytesReference.fromByteBuffer(byteBuffer).streamInput()) {
            int size = streamInput.readInt();
            for (int i=0; i<size; i++) {
                ModelTensors tensorOutput = new ModelTensors(streamInput);
                tensorOutput.filter(resultFilter);
                tensorOutputs.add(tensorOutput);
            }
        } catch (Exception e) {
            log.error("Failed to parse output", e);
        }
    }

}
