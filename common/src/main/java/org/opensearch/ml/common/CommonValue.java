/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common;

import static org.opensearch.ml.common.model.MLModelConfig.ALL_CONFIG_FIELD;
import static org.opensearch.ml.common.model.MLModelConfig.MODEL_TYPE_FIELD;
import static org.opensearch.ml.common.model.TextEmbeddingModelConfig.EMBEDDING_DIMENSION_FIELD;
import static org.opensearch.ml.common.model.TextEmbeddingModelConfig.FRAMEWORK_TYPE_FIELD;
import static org.opensearch.ml.common.model.TextEmbeddingModelConfig.MODEL_MAX_LENGTH_FIELD;
import static org.opensearch.ml.common.model.TextEmbeddingModelConfig.NORMALIZE_RESULT_FIELD;
import static org.opensearch.ml.common.model.TextEmbeddingModelConfig.POOLING_MODE_FIELD;

public class CommonValue {

    public static Integer NO_SCHEMA_VERSION = 0;
    public static final String USER = "user";
    public static final String META = "_meta";
    public static final String SCHEMA_VERSION_FIELD = "schema_version";
    public static final String UNLOADED = "unloaded";
    public static final String NOT_FOUND = "not_found";

    public static final String BOX_TYPE_KEY = "box_type";
    //hot node
    public static String HOT_BOX_TYPE = "hot";
    // warm node
    public static String WARM_BOX_TYPE = "warm";

    public static final String ML_MODEL_INDEX = ".plugins-ml-model";
    public static final String ML_TASK_INDEX = ".plugins-ml-task";
    public static final Integer ML_MODEL_INDEX_SCHEMA_VERSION = 3;
    public static final Integer ML_TASK_INDEX_SCHEMA_VERSION = 1;
    public static final String USER_FIELD_MAPPING = "      \""
            + CommonValue.USER
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
            + MLModel.ALGORITHM_FIELD
            + "\": {\"type\": \"keyword\"},\n"
            + "      \""
            + MLModel.MODEL_NAME_FIELD
            + "\" : {\"type\":\"text\",\"fields\":{\"keyword\":{\"type\":\"keyword\",\"ignore_above\":256}}},\n"
            + "      \""
            + MLModel.OLD_MODEL_VERSION_FIELD
            + "\" : {\"type\": \"long\"},\n"
            + "      \""
            + MLModel.MODEL_VERSION_FIELD
            + "\" : {\"type\": \"keyword\"},\n"
            + "      \""
            + MLModel.MODEL_CONTENT_FIELD
            + "\" : {\"type\": \"binary\"},\n"
            + "      \""
            + MLModel.CHUNK_NUMBER_FIELD
            + "\" : {\"type\": \"long\"},\n"
            + "      \""
            + MLModel.TOTAL_CHUNKS_FIELD
            + "\" : {\"type\": \"long\"},\n"
            + "      \""
            + MLModel.MODEL_ID_FIELD
            + "\" : {\"type\": \"keyword\"},\n"
            + "      \""
            + MLModel.DESCRIPTION_FIELD
            + "\" : {\"type\": \"text\"},\n"
            + "      \""
            + MLModel.MODEL_FORMAT_FIELD
            + "\" : {\"type\": \"keyword\"},\n"
            + "      \""
            + MLModel.MODEL_STATE_FIELD
            + "\" : {\"type\": \"keyword\"},\n"
            + "      \""
            + MLModel.MODEL_CONTENT_SIZE_IN_BYTES_FIELD
            + "\" : {\"type\": \"long\"},\n"
            + "      \""
            + MLModel.PLANNING_WORKER_NODE_COUNT_FIELD
            + "\" : {\"type\": \"integer\"},\n"
            + "      \""
            + MLModel.CURRENT_WORKER_NODE_COUNT_FIELD
            + "\" : {\"type\": \"integer\"},\n"
            + "      \""
            + MLModel.MODEL_CONFIG_FIELD
            + "\" : {\"properties\":{\""
            + MODEL_TYPE_FIELD + "\":{\"type\":\"keyword\"},\""
            + EMBEDDING_DIMENSION_FIELD + "\":{\"type\":\"integer\"},\""
            + FRAMEWORK_TYPE_FIELD + "\":{\"type\":\"keyword\"},\""
            + POOLING_MODE_FIELD + "\":{\"type\":\"keyword\"},\""
            + NORMALIZE_RESULT_FIELD + "\":{\"type\":\"boolean\"},\""
            + MODEL_MAX_LENGTH_FIELD + "\":{\"type\":\"integer\"},\""
            + ALL_CONFIG_FIELD + "\":{\"type\":\"text\"}}},\n"
            + "      \""
            + MLModel.MODEL_CONTENT_HASH_VALUE_FIELD
            + "\" : {\"type\": \"keyword\"},\n"
            + "      \""
            + MLModel.CREATED_TIME_FIELD
            + "\": {\"type\": \"date\", \"format\": \"strict_date_time||epoch_millis\"},\n"
            + "      \""
            + MLModel.LAST_UPDATED_TIME_FIELD
            + "\": {\"type\": \"date\", \"format\": \"strict_date_time||epoch_millis\"},\n"
            + "      \""
            + MLModel.LAST_UPLOADED_TIME_FIELD
            + "\": {\"type\": \"date\", \"format\": \"strict_date_time||epoch_millis\"},\n"
            + "      \""
            + MLModel.LAST_LOADED_TIME_FIELD
            + "\": {\"type\": \"date\", \"format\": \"strict_date_time||epoch_millis\"},\n"
            + "      \""
            + MLModel.LAST_UNLOADED_TIME_FIELD
            + "\": {\"type\": \"date\", \"format\": \"strict_date_time||epoch_millis\"},\n"
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
}
