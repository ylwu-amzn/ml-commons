/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.helper;

import static org.opensearch.ml.common.CommonValue.CONNECTOR_ACTION_FIELD;
import static org.opensearch.ml.common.CommonValue.ML_LONG_MEMORY_HISTORY_INDEX_MAPPING_PATH;
import static org.opensearch.ml.common.CommonValue.ML_LONG_TERM_MEMORY_INDEX_MAPPING_PATH;
import static org.opensearch.ml.common.CommonValue.ML_MEMORY_SESSION_INDEX_MAPPING_PATH;
import static org.opensearch.ml.common.CommonValue.ML_WORKING_MEMORY_INDEX_MAPPING_PATH;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.KNN_EF_CONSTRUCTION;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.KNN_ENGINE;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.KNN_M;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.KNN_METHOD_NAME;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.KNN_SPACE_TYPE;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.MEMORY_EMBEDDING_FIELD;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.remote.RemoteInferenceMLInput;
import org.opensearch.ml.common.memorycontainer.MemoryConfiguration;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.transport.connector.MLExecuteConnectorAction;
import org.opensearch.ml.common.transport.connector.MLExecuteConnectorRequest;
import org.opensearch.ml.common.utils.StringUtils;
import org.opensearch.ml.engine.indices.MLIndicesHandler;
import org.opensearch.transport.client.Client;

import lombok.extern.log4j.Log4j2;

/**
 * Helper class for creating memory indices in remote storage using connectors
 */
@Log4j2
public class RemoteStorageHelper {

    private static final String CREATE_INDEX_ACTION = "create_index";
    private static final String INDEX_NAME_PARAM = "index_name";
    private static final String INPUT_PARAM = "input";

    /**
     * Creates a memory index in remote storage using a connector
     *
     * @param connectorId The connector ID to use for remote storage
     * @param indexName The name of the index to create
     * @param indexMapping The index mapping as a JSON string
     * @param client The OpenSearch client
     * @param listener The action listener
     */
    public static void createRemoteIndex(
        String connectorId,
        String indexName,
        String indexMapping,
        Client client,
        ActionListener<Boolean> listener
    ) {
        try {
            // Parse the mapping string to a Map
            Map<String, Object> mappingMap = parseMappingToMap(indexMapping);

            // Build the request body for creating the index
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("mappings", mappingMap);

            // Prepare parameters for connector execution
            Map<String, String> parameters = new HashMap<>();
            parameters.put(INDEX_NAME_PARAM, indexName);
            parameters.put(INPUT_PARAM, StringUtils.toJson(requestBody));
            parameters.put(CONNECTOR_ACTION_FIELD, CREATE_INDEX_ACTION);

            // Execute the connector action
            executeConnectorAction(connectorId, parameters, client, ActionListener.wrap(response -> {
                log.info("Successfully created remote index: {}", indexName);
                listener.onResponse(true);
            }, e -> {
                log.error("Failed to create remote index: {}", indexName, e);
                listener.onFailure(e);
            }));

        } catch (Exception e) {
            log.error("Error preparing remote index creation for: {}", indexName, e);
            listener.onFailure(e);
        }
    }

    /**
     * Creates session memory index in remote storage
     */
    public static void createRemoteSessionMemoryIndex(
        String connectorId,
        String indexName,
        MemoryConfiguration configuration,
        MLIndicesHandler mlIndicesHandler,
        Client client,
        ActionListener<Boolean> listener
    ) {
        String indexMappings = mlIndicesHandler.getMapping(ML_MEMORY_SESSION_INDEX_MAPPING_PATH);
        createRemoteIndex(connectorId, indexName, indexMappings, client, listener);
    }

    /**
     * Creates working memory index in remote storage
     */
    public static void createRemoteWorkingMemoryIndex(
        String connectorId,
        String indexName,
        MemoryConfiguration configuration,
        MLIndicesHandler mlIndicesHandler,
        Client client,
        ActionListener<Boolean> listener
    ) {
        String indexMappings = mlIndicesHandler.getMapping(ML_WORKING_MEMORY_INDEX_MAPPING_PATH);
        createRemoteIndex(connectorId, indexName, indexMappings, client, listener);
    }

    /**
     * Creates long-term memory history index in remote storage
     */
    public static void createRemoteLongTermMemoryHistoryIndex(
        String connectorId,
        String indexName,
        MemoryConfiguration configuration,
        MLIndicesHandler mlIndicesHandler,
        Client client,
        ActionListener<Boolean> listener
    ) {
        String indexMappings = mlIndicesHandler.getMapping(ML_LONG_MEMORY_HISTORY_INDEX_MAPPING_PATH);
        createRemoteIndex(connectorId, indexName, indexMappings, client, listener);
    }

    /**
     * Creates long-term memory index in remote storage with dynamic embedding configuration
     */
    public static void createRemoteLongTermMemoryIndex(
        String connectorId,
        String indexName,
        MemoryConfiguration memoryConfig,
        MLIndicesHandler mlIndicesHandler,
        Client client,
        ActionListener<Boolean> listener
    ) {
        try {
            String indexMapping = buildLongTermMemoryMapping(memoryConfig, mlIndicesHandler);
            createRemoteIndex(connectorId, indexName, indexMapping, client, listener);
        } catch (Exception e) {
            log.error("Failed to build long-term memory mapping for remote index: {}", indexName, e);
            listener.onFailure(e);
        }
    }

    /**
     * Builds the long-term memory index mapping dynamically based on configuration
     */
    private static String buildLongTermMemoryMapping(MemoryConfiguration memoryConfig, MLIndicesHandler mlIndicesHandler)
        throws IOException {
        String baseMappingJson = mlIndicesHandler.getMapping(ML_LONG_TERM_MEMORY_INDEX_MAPPING_PATH);

        Map<String, Object> mapping = new HashMap<>();
        Map<String, Object> properties = new HashMap<>();

        XContentParser parser = XContentHelper
            .createParser(
                NamedXContentRegistry.EMPTY,
                LoggingDeprecationHandler.INSTANCE,
                new BytesArray(baseMappingJson),
                XContentType.JSON
            );

        Map<String, Object> baseMapping = parser.mapOrdered();
        mapping.put("_meta", baseMapping.get("_meta"));
        properties.putAll((Map<String, Object>) baseMapping.get("properties"));

        // Add embedding field based on configuration
        if (memoryConfig.getEmbeddingModelType() == FunctionName.TEXT_EMBEDDING) {
            Map<String, Object> knnVector = new HashMap<>();
            knnVector.put("type", "knn_vector");
            knnVector.put("dimension", memoryConfig.getDimension());

            Map<String, Object> method = new HashMap<>();
            method.put("name", KNN_METHOD_NAME);
            method.put("space_type", KNN_SPACE_TYPE);
            method.put("engine", KNN_ENGINE);
            method.put("parameters", Map.of("ef_construction", KNN_EF_CONSTRUCTION, "m", KNN_M));
            knnVector.put("method", method);

            properties.put(MEMORY_EMBEDDING_FIELD, knnVector);
        } else if (memoryConfig.getEmbeddingModelType() == FunctionName.SPARSE_ENCODING) {
            properties.put(MEMORY_EMBEDDING_FIELD, Map.of("type", "rank_features"));
        }

        mapping.put("properties", properties);
        return StringUtils.toJson(mapping);
    }

    /**
     * Executes a connector action
     */
    private static void executeConnectorAction(
        String connectorId,
        Map<String, String> parameters,
        Client client,
        ActionListener<ModelTensorOutput> listener
    ) {
        RemoteInferenceInputDataSet inputDataSet = RemoteInferenceInputDataSet.builder().parameters(parameters).build();
        MLInput mlInput = RemoteInferenceMLInput.builder().algorithm(FunctionName.CONNECTOR).inputDataset(inputDataSet).build();
        MLExecuteConnectorRequest request = new MLExecuteConnectorRequest(connectorId, mlInput);

        client.execute(MLExecuteConnectorAction.INSTANCE, request, ActionListener.wrap(r -> {
            ModelTensorOutput output = (ModelTensorOutput) r.getOutput();
            listener.onResponse(output);
        }, e -> {
            log.error("Failed to execute connector action for connector: {}", connectorId, e);
            listener.onFailure(e);
        }));
    }

    /**
     * Parses a JSON mapping string to a Map
     */
    private static Map<String, Object> parseMappingToMap(String mappingJson) throws IOException {
        XContentParser parser = XContentHelper
            .createParser(NamedXContentRegistry.EMPTY, LoggingDeprecationHandler.INSTANCE, new BytesArray(mappingJson), XContentType.JSON);
        return parser.mapOrdered();
    }
}
