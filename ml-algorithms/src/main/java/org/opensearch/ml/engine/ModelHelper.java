/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.DownloadUtils;
import ai.djl.training.util.ProgressBar;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hashing;
import com.google.common.io.ByteSource;
import com.google.gson.Gson;
import com.google.gson.stream.JsonReader;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FileUtils;
import org.opensearch.action.ActionListener;
import org.opensearch.ml.common.model.MLModelConfig;
import org.opensearch.ml.common.model.MLModelFormat;
import org.opensearch.ml.common.model.TextEmbeddingModelConfig;
import org.opensearch.ml.common.transport.upload.MLUploadInput;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.security.AccessController;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import static org.opensearch.ml.engine.MLEngine.getModelCachePath;
import static org.opensearch.ml.engine.MLEngine.getLoadModelPath;
import static org.opensearch.ml.engine.MLEngine.getLocalPrebuiltModelConfigPath;
import static org.opensearch.ml.engine.MLEngine.getLocalPrebuiltModelPath;
import static org.opensearch.ml.engine.MLEngine.getUploadModelPath;
import static org.opensearch.ml.engine.utils.FileUtils.deleteFileQuietly;
import static org.opensearch.ml.engine.utils.FileUtils.readAndFragment;

@Log4j2
public class ModelHelper {
    public static final String CHUNK_FILES = "chunk_files";
    public static final String MODEL_SIZE_IN_BYTES = "model_size_in_bytes";
    public static final String MODEL_FILE_HASH = "model_file_hash";
    private static final int CHUNK_SIZE = 10_000_000; // 10MB
    public static final String PT = ".pt";
    public static final String ONNX = ".onnx";
    public static final String PYTORCH_ENGINE = "PyTorch";
    public static final String ONNX_ENGINE = "OnnxRuntime";

    private Map<String, Predictor> predictors;
    private Map<String, TextEmbeddingModelConfig.FrameworkType> modelTransformersTypes;
    private Map<String, ZooModel> models;
    private Gson gson;

    public ModelHelper() {
        modelTransformersTypes = new ConcurrentHashMap<>();
        predictors = new ConcurrentHashMap<>();
        models = new ConcurrentHashMap<>();
        gson = new Gson();
    }

    public void downloadPrebuiltModelConfig(String taskId, MLUploadInput uploadInput, ActionListener<MLUploadInput> listener) {
        String modelName = uploadInput.getModelName();
        Integer version = uploadInput.getVersion();
        boolean loadModel = uploadInput.isLoadModel();
        String[] modelNodeIds = uploadInput.getModelNodeIds();
        try {
            AccessController.doPrivileged((PrivilegedExceptionAction<Void>) () -> {

                Path modelUploadPath = getUploadModelPath(taskId, modelName, version);
                String configCacheFilePath = modelUploadPath.resolve("config.json").toString();

                String localConfigFile = getLocalPrebuiltModelConfigPath(modelName, version).toString();
                String configFileUrl = "file://" + localConfigFile;
                String modelZipFileUrl = "file://" + getLocalPrebuiltModelPath(modelName, version);
                DownloadUtils.download(configFileUrl, configCacheFilePath, new ProgressBar());


                Map<?, ?> config = null;
                try (JsonReader reader = new JsonReader(new FileReader(localConfigFile))) {
                    config = gson.fromJson(reader, Map.class);
                }

                if (config == null) {
                    listener.onFailure(new IllegalArgumentException("model config not found"));
                    return null;
                }

                MLUploadInput.MLUploadInputBuilder builder = MLUploadInput.builder();

                builder.modelName(modelName).version(version).url(modelZipFileUrl).loadModel(loadModel).modelNodeIds(modelNodeIds);
                config.entrySet().forEach(entry -> {
                    switch (entry.getKey().toString()) {
                        case MLUploadInput.MODEL_FORMAT_FIELD:
                            builder.modelFormat(MLModelFormat.from(entry.getValue().toString()));
                            break;
                        case MLUploadInput.MODEL_CONFIG_FIELD:
                            TextEmbeddingModelConfig.TextEmbeddingModelConfigBuilder configBuilder = TextEmbeddingModelConfig.builder();
                            Map<?, ?> configMap = (Map<?, ?>) entry.getValue();
                            for (Map.Entry<?, ?> configEntry : configMap.entrySet()) {
                                switch (configEntry.getKey().toString()) {
                                    case MLModelConfig.MODEL_TYPE_FIELD:
                                        configBuilder.modelType(configEntry.getValue().toString());
                                        break;
                                    case MLModelConfig.ALL_CONFIG_FIELD:
                                        configBuilder.allConfig(configEntry.getValue().toString());
                                        break;
                                    case TextEmbeddingModelConfig.EMBEDDING_DIMENSION_FIELD:
                                        configBuilder.embeddingDimension(((Double)configEntry.getValue()).intValue());
                                        break;
                                    case TextEmbeddingModelConfig.FRAMEWORK_TYPE_FIELD:
                                        configBuilder.frameworkType(TextEmbeddingModelConfig.FrameworkType.from(configEntry.getValue().toString()));
                                        break;
                                    default:
                                        break;
                                }
                            }
                            builder.modelConfig(configBuilder.build());
                            break;
                        default:
                            break;
                    }
                });
                MLUploadInput mlUploadInput = builder.build();
                listener.onResponse(mlUploadInput);
                return null;
            });
        } catch (Exception e) {
            listener.onFailure(e);
        } finally {
            deleteFileQuietly(getUploadModelPath(taskId));
        }
    }

    public void downloadAndSplit(String modelId, String modelName, Integer version, String url, ActionListener<Map<String, Object>> listener) {
        try {
            AccessController.doPrivileged((PrivilegedExceptionAction<Void>) () -> {
                Path modelUploadPath = getUploadModelPath(modelId, modelName, version);
                String modelPath = modelUploadPath +".zip";
                Path modelPartsPath = modelUploadPath.resolve("chunks");
                File modelZipFile = new File(modelPath);
                DownloadUtils.download(url, modelPath, new ProgressBar());
                verifyModelZipFile(modelPath);

                ArrayList<String> chunkFiles = readAndFragment(modelZipFile, modelPartsPath, CHUNK_SIZE);
                Map<String, Object> result = new HashMap<>();
                result.put(CHUNK_FILES, chunkFiles);
                result.put(MODEL_SIZE_IN_BYTES, modelZipFile.length());

                result.put(MODEL_FILE_HASH, calculateFileHash(modelZipFile));
                FileUtils.delete(modelZipFile);
                listener.onResponse(result);
                return null;
            });
        } catch (Exception e) {
            listener.onFailure(e);
        }
    }

    public String calculateFileHash(File modelZipFile) throws IOException {
        ByteSource byteSource = com.google.common.io.Files.asByteSource(modelZipFile);
        HashCode hc = byteSource.hash(Hashing.sha256());
        return hc.toString();
    }

    //TODO: check if model is ONNX or torchscript with content
    private void verifyModelZipFile(String modelZipFilePath) throws IOException {
        boolean hasModelFile = false;
        boolean hasTokenizerFile = false;
        try (ZipFile zipFile = new ZipFile(modelZipFilePath)) {
            Enumeration zipEntries = zipFile.entries();
            while (zipEntries.hasMoreElements()) {
                String fileName = ((ZipEntry) zipEntries.nextElement()).getName();
                if (fileName.endsWith(".pt") || fileName.endsWith(".onnx")) {
                    if (hasModelFile) {
                        throw new IllegalArgumentException("Find multiple model files");
                    }
                    hasModelFile = true;
                }
                if (fileName.equals("tokenizer.json")) {
                    if (hasTokenizerFile) {
                        throw new IllegalArgumentException("Find tokenizer files");
                    }
                    hasTokenizerFile = true;
                }
            }
        }
        if (!hasModelFile) {
            throw new IllegalArgumentException("Can't find model file");
        }
        if (!hasTokenizerFile) {
            throw new IllegalArgumentException("Can't find tokenizer file");
        }
    }


    public void unloadModel(String modelId) {
        deleteFileCache(modelId);
        if (predictors.containsKey(modelId)) {
            log.debug("unload mode: close and remove predictor {}", modelId);
            predictors.get(modelId).close();
            predictors.remove(modelId);
        }
        if (models.containsKey(modelId)) {
            log.debug("unload mode: close and remove model {}", modelId);
            models.get(modelId).close();
            models.remove(modelId);
        }
    }

    public void deleteFileCache(String modelId) {
        deleteFileQuietly(getModelCachePath(modelId));
        deleteFileQuietly(getLoadModelPath(modelId));
        deleteFileQuietly(getUploadModelPath(modelId));
    }

}
