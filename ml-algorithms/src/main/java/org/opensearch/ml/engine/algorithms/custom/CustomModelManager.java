/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.custom;

import ai.djl.training.util.DownloadUtils;
import ai.djl.training.util.ProgressBar;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hashing;
import com.google.common.io.ByteSource;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FileUtils;
import org.opensearch.action.ActionListener;

import java.io.File;
import java.nio.file.Path;
import java.security.AccessController;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static org.opensearch.ml.engine.MLEngine.getUploadModelPath;
import static org.opensearch.ml.engine.utils.MLFileUtils.readAndFragment;

@Log4j2
public class CustomModelManager {
    public static final String CHUNK_FILES = "chunk_files";
    public static final String MODEL_SIZE_IN_BYTES = "model_size_in_bytes";
    public static final String MODEL_FILE_MD5 = "model_file_md5";
    private static final int CHUNK_SIZE = 10_000_000; // 10MB

    public CustomModelManager() {}

    public void downloadAndSplit(String modelId, String modelName, Integer version, String url, ActionListener<Map<String, Object>> listener) {
        try {
            log.info("Download model zip and split into chunks for modelId:{}, modelName:{}, version:{}", modelId, modelName, version);
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
                result.put(MODEL_FILE_MD5, hc.toString());
                FileUtils.delete(modelZipFile);
                listener.onResponse(result);
                return null;
            });
        } catch (Exception e) {
            log.error("Failed to download model file and split for " + modelId, e);
            listener.onFailure(e);
        }
    }

}
