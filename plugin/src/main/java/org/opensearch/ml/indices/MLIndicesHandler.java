/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.indices;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.action.admin.indices.create.CreateIndexRequest;
import org.opensearch.action.admin.indices.create.CreateIndexResponse;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.ml.common.CommonName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.MLTask;

@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
@RequiredArgsConstructor
@Log4j2
public class MLIndicesHandler {
    public static final String ML_MODEL_INDEX = ".plugins-ml-model";
    public static final String ML_TASK_INDEX = ".plugins-ml-task";
    public static final Integer ML_MODEL_INDEX_SCHEMA_VERSION = 1;
    public static final Integer ML_TASK_INDEX_SCHEMA_VERSION = 1;
    public static final String USER_FIELD_MAPPING = "      \""
        + CommonName.USER
        + "\": {\n"
        + "        \"type\": \"nested\",\n"
        + "        \"properties\": {\n"
        + "          \"name\": {\"type\":\"text\", \"fields\":{\"keyword\":{\"type\":\"keyword\", \"ignore_above\":256}}},\n"
        + "          \"backend_roles\": {\"type\":\"text\", \"fields\":{\"keyword\":{\"type\":\"keyword\"}}},\n"
        + "          \"roles\": {\"type\":\"text\", \"fields\":{\"keyword\":{\"type\":\"keyword\"}}},\n"
        + "          \"custom_attribute_names\": {\"type\":\"text\", \"fields\":{\"keyword\":{\"type\":\"keyword\"}}}\n"
        + "        }\n"
        + "      }\n";
    public static final String ML_MODEL_INDEX_MAPPING = "{\n"
        + "    \"_meta\": {\"schema_version\": "
        + ML_MODEL_INDEX_SCHEMA_VERSION
        + "},\n"
        + "    \"properties\": {\n"
        + "      \""
        + MLModel.ALGORITHM
        + "\": {\"type\": \"keyword\"},\n"
        + "      \""
        + MLModel.MODEL_NAME
        + "\" : {\"type\": \"keyword\"},\n"
        + "      \""
        + MLModel.MODEL_VERSION
        + "\" : {\"type\": \"keyword\"},\n"
        + "      \""
        + MLModel.MODEL_CONTENT
        + "\" : {\"type\": \"binary\"},\n"
        + USER_FIELD_MAPPING
        + "    }\n"
        + "}";

    public static final String ML_TASK_INDEX_MAPPING = "{\n"
        + "    \"_meta\": {\"schema_version\": "
        + ML_TASK_INDEX_SCHEMA_VERSION
        + "},\n"
        + "    \"properties\": {\n"
        + "      \""
        + MLTask.MODEL_ID_FIELD
        + "\": {\"type\": \"keyword\"},\n"
        + "      \""
        + MLTask.TASK_TYPE_FIELD
        + "\": {\"type\": \"keyword\"},\n"
        + "      \""
        + MLTask.FUNCTION_NAME_FIELD
        + "\": {\"type\": \"keyword\"},\n"
        + "      \""
        + MLTask.STATE_FIELD
        + "\": {\"type\": \"keyword\"},\n"
        + "      \""
        + MLTask.INPUT_TYPE_FIELD
        + "\": {\"type\": \"keyword\"},\n"
        + "      \""
        + MLTask.PROGRESS_FIELD
        + "\": {\"type\": \"float\"},\n"
        + "      \""
        + MLTask.OUTPUT_INDEX_FIELD
        + "\": {\"type\": \"keyword\"},\n"
        + "      \""
        + MLTask.WORKER_NODE_FIELD
        + "\": {\"type\": \"keyword\"},\n"
        + "      \""
        + MLTask.CREATE_TIME_FIELD
        + "\": {\"type\": \"date\", \"format\": \"strict_date_time||epoch_millis\"},\n"
        + "      \""
        + MLTask.LAST_UPDATE_TIME_FIELD
        + "\": {\"type\": \"date\", \"format\": \"strict_date_time||epoch_millis\"},\n"
        + "      \""
        + MLTask.ERROR_FIELD
        + "\": {\"type\": \"text\"},\n"
        + "      \""
        + MLTask.IS_ASYNC_TASK_FIELD
        + "\" : {\"type\" : \"boolean\"}, \n"
        + USER_FIELD_MAPPING
        + "    }\n"
        + "}";

    ClusterService clusterService;
    Client client;

    private static final Map<String, AtomicBoolean> indexMappingUpdated = new HashMap<>();

    public void initModelIndexIfAbsent(ActionListener<Boolean> listener) {
        initMLIndexIfAbsent(ML_MODEL_INDEX, ML_MODEL_INDEX_MAPPING, listener);
    }

    public void initMLTaskIndex(ActionListener<Boolean> listener) {
        initMLIndexIfAbsent(ML_TASK_INDEX, ML_TASK_INDEX_MAPPING, listener);
    }

    public void initMLIndexIfAbsent(String indexName, String mapping, ActionListener<Boolean> listener) {
        if (!clusterService.state().metadata().hasIndex(indexName)) {
            try (ThreadContext.StoredContext threadContext = client.threadPool().getThreadContext().stashContext()) {
                ActionListener<CreateIndexResponse> actionListener = ActionListener.wrap(r -> {
                    if (r.isAcknowledged()) {
                        log.info("create index:{}", indexName);
                        listener.onResponse(true);
                    } else {
                        listener.onResponse(false);
                    }
                }, e -> {
                    log.error("Failed to create index " + indexName, e);
                    listener.onFailure(e);
                });
                CreateIndexRequest request = new CreateIndexRequest(indexName).mapping(mapping);
                client.admin().indices().create(request, ActionListener.runBefore(actionListener, () -> threadContext.restore()));
            } catch (Exception e) {
                log.error("Failed to init index " + indexName, e);
                listener.onFailure(e);
            }
        } else {
            log.info("index:{} is already created", indexName);
            // if (indexMappingUpdated) {
            //
            // }
            listener.onResponse(true);
        }
    }

}
