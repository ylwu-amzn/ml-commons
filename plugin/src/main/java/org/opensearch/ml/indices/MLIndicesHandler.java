/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.indices;

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

@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
@RequiredArgsConstructor
@Log4j2
public class MLIndicesHandler {
    public static final String ML_MODEL_INDEX = ".plugins-ml-model";
    public static final String ML_TASK_INDEX = ".plugins-ml-task";
    private static final String ML_MODEL_INDEX_MAPPING = "{\n"
        + "    \"properties\": {\n"
        + "      \"task_id\": { \"type\": \"keyword\" },\n"
        + "      \"algorithm\": {\"type\": \"keyword\"},\n"
        + "      \"name\" : { \"type\": \"keyword\"},\n"
        + "      \"version\" : { \"type\": \"keyword\"},\n"
        + "      \"content\" : { \"type\": \"binary\"}\n"
        + "    }\n"
        + "}";

    private static final String ML_TASK_INDEX_MAPPING = "{\n"
        + "    \"properties\": {\n"
        + "      \"model_id\": {\"type\": \"keyword\"},\n"
        + "      \"task_type\": {\"type\": \"keyword\"},\n"
        + "      \"function_name\": {\"type\": \"keyword\"},\n"
        + "      \"state\": {\"type\": \"keyword\"},\n"
        + "      \"input_type\": {\"type\": \"keyword\"},\n"
        + "      \"progress\": {\"type\": \"float\"},\n"
        + "      \"output_index\": {\"type\": \"keyword\"},\n"
        + "      \"worker_node\": {\"type\": \"keyword\"},\n"
        + "      \"create_time\": {\"type\": \"date\", \"format\": \"strict_date_time||epoch_millis\"},\n"
        + "      \"last_update_time\": {\"type\": \"date\", \"format\": \"strict_date_time||epoch_millis\"},\n"
        + "      \"error\": {\"type\": \"text\"},\n"
        + "      \"user\": {\n"
        + "        \"type\": \"nested\",\n"
        + "        \"properties\": {\n"
        + "          \"name\": {\"type\":\"text\", \"fields\":{\"keyword\":{\"type\":\"keyword\", \"ignore_above\":256}}},\n"
        + "          \"backend_roles\": {\"type\":\"text\", \"fields\":{\"keyword\":{\"type\":\"keyword\"}}},\n"
        + "          \"roles\": {\"type\":\"text\", \"fields\":{\"keyword\":{\"type\":\"keyword\"}}},\n"
        + "          \"custom_attribute_names\": {\"type\":\"text\", \"fields\":{\"keyword\":{\"type\":\"keyword\"}}}\n"
        + "        }\n"
        + "      }\n"
        + "    }\n"
        + "}";

    ClusterService clusterService;
    Client client;

    public void initModelIndexIfAbsent() {
        initMLIndexIfAbsent(ML_MODEL_INDEX, ML_MODEL_INDEX_MAPPING);
    }

    public boolean doesModelIndexExist() {
        return clusterService.state().metadata().hasIndex(ML_MODEL_INDEX);
    }

    private void initMLIndexIfAbsent(String indexName, String mapping) {
        if (!clusterService.state().metadata().hasIndex(indexName)) {
            client.admin().indices().prepareCreate(indexName).get();
            log.info("create index:{}", indexName);
        } else {
            log.info("index:{} is already created", indexName);
        }
    }

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
            listener.onResponse(true);
        }
    }

}
