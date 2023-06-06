/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.connector;

import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.text.StringSubstitutor;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.core.xcontent.XContentParser;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;

import static org.opensearch.ml.common.connector.ConnectorNames.CHAT_V1;

@Log4j2
@NoArgsConstructor
@org.opensearch.ml.common.annotation.Connector(CHAT_V1)
public class ChatConnector extends HttpConnector {
    public static final String CONTENT_INDEX = "content_index";
    public static final String EMBEDDING_MODEL_ID_FIELD = "embedding_model_id";
    public static final String EMBEDDING_FIELD_FIELD = "embedding_field";
    public static final String CONTENT_FIELD_FIELD = "content_fields";
    public static final String CONTENT_DOC_SIZE_FIELD = "content_doc_size";
    private static final Integer DEFAULT_CONTENT_DOC_SIZE = 2;
    public static final String SESSION_INDEX_FIELD = "session_index";
    public static final String SESSION_ID_FIELD_FIELD = "session_id_field";
    public static final String SESSION_SIZE_FIELD = "session_size";
    private static final Integer DEFAULT_SESSION_SIZE = 2;

    private static final String DEFAULT_SEMANTIC_SEARCH_QUERY = "{\n" +
            "  \"query\": {\n" +
            "    \"neural\": {\n" +
            "      \"${parameters.embedding_field}\": {\n" +
            "        \"query_text\": \"${parameters.question}\",\n" +
            "        \"model_id\": \"${parameters.embedding_model_id}\",\n" +
            "        \"k\": 100\n" +
            "      }\n" +
            "    }\n" +
            "  },\n" +
            "  \"size\": \"${parameters.content_doc_size}\",\n" +
            "  \"_source\": [\n" +
            "    \"${parameters.content_fields}\"\n" +
            "  ]\n" +
            "}";

    public static final String DEFAULT_SEMANTIC_SEARCH_QUERY_FOR_SESSION = "{\n" +
            "  \"query\": {\n" +
            "    \"bool\": {\n" +
            "      \"should\": [\n" +
            "        {\n" +
            "          \"neural\": {\n" +
            "            \"question_knn\": {\n" +
            "              \"query_text\": \"${parameters.question}\",\n" +
            "              \"model_id\": \"${parameters.session_index_embedding_model_id}\",\n" +
            "              \"k\": 10\n" +
            "            }\n" +
            "          }\n" +
            "        },\n" +
            "        {\n" +
            "          \"neural\": {\n" +
            "            \"answer_knn\": {\n" +
            "              \"query_text\": \"${parameters.question}\",\n" +
            "              \"model_id\": \"${parameters.session_index_embedding_model_id}\",\n" +
            "              \"k\": 10\n" +
            "            }\n" +
            "          }\n" +
            "        }\n" +
            "      ],\n" +
            "      \"must\": [\n" +
            "         {\n" +
            "           \"term\": {\n" +
            "             \"session_id\": {\n" +
            "               \"value\": \"${parameters.session_id}\"\n" +
            "             }\n" +
            "           }\n" +
            "         }\n" +
            "      ]\n" +
            "    }\n" +
            "  },\n" +
            "  \"size\": \"${parameters.session_size}\",\n" +
            "  \"_source\": [\n" +
            "    \"question\",\n" +
            "    \"answer\"\n" +
            "  ]\n" +
            "}";

    public static final String DEFAULT_SEMANTIC_SEARCH_QUERY_FOR_ALL_SESSIONS = "{\n" +
            "  \"query\": {\n" +
            "    \"bool\": {\n" +
            "      \"should\": [\n" +
            "        {\n" +
            "          \"neural\": {\n" +
            "            \"question_knn\": {\n" +
            "              \"query_text\": \"${parameters.question}\",\n" +
            "              \"model_id\": \"${parameters.session_index_embedding_model_id}\",\n" +
            "              \"k\": 10\n" +
            "            }\n" +
            "          }\n" +
            "        },\n" +
            "        {\n" +
            "          \"neural\": {\n" +
            "            \"answer_knn\": {\n" +
            "              \"query_text\": \"${parameters.question}\",\n" +
            "              \"model_id\": \"${parameters.session_index_embedding_model_id}\",\n" +
            "              \"k\": 10\n" +
            "            }\n" +
            "          }\n" +
            "        }\n" +
            "      ]\n" +
            "    }\n" +
            "  },\n" +
            "  \"size\": \"${parameters.session_size}\",\n" +
            "  \"_source\": [\n" +
            "    \"question\",\n" +
            "    \"answer\"\n" +
            "  ]\n" +
            "}";

    public ChatConnector(String name, XContentParser parser) throws IOException {
        super(name, parser);
        validate();
    }

    public ChatConnector(StreamInput input) throws IOException {
        super(input);
        validate();
    }

    private void validate() {
        if (credential == null) {
            throw new IllegalArgumentException("Missing credential");
        }
    }

    public String createNeuralSearchQuery(Map<String, String> params) {
        String searchTemplate = params.containsKey("search_query") ? params.get("search_query") : DEFAULT_SEMANTIC_SEARCH_QUERY;
        StringSubstitutor substitutor = new StringSubstitutor(params, "${parameters.", "}");
        String query = substitutor.replace(searchTemplate);
        return query;
    }

    public String createNeuralSearchQueryForSession(Map<String, String> params) {
        String withAllSessions = params.get("with_all_sessions");
        boolean searchAllSessions = withAllSessions != null && "true".equals(withAllSessions.toLowerCase(Locale.ROOT));
        String searchTemplate = searchAllSessions? DEFAULT_SEMANTIC_SEARCH_QUERY_FOR_ALL_SESSIONS : DEFAULT_SEMANTIC_SEARCH_QUERY_FOR_SESSION;
        StringSubstitutor substitutor = new StringSubstitutor(params, "${parameters.", "}");
        String query = substitutor.replace(searchTemplate);
        return query;
    }

    @Override
    public Connector clone() {
        try (BytesStreamOutput bytesStreamOutput = new BytesStreamOutput()){
            this.writeTo(bytesStreamOutput);
            StreamInput streamInput = bytesStreamOutput.bytes().streamInput();
            return new ChatConnector(streamInput);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public String getContentIndex() {
        return parameters.get(CONTENT_INDEX);
    }

    public String getEmbeddingModel() {
        return parameters.get(EMBEDDING_MODEL_ID_FIELD);
    }

    public String getEmbeddingField() {
        return parameters.get(EMBEDDING_FIELD_FIELD);
    }

    public String getContentFields() {
        return parameters.get(CONTENT_FIELD_FIELD);
    }

    public Integer getContentDocSize() {
        if (!parameters.containsKey(CONTENT_DOC_SIZE_FIELD)) {
            return DEFAULT_CONTENT_DOC_SIZE;
        }
        return Integer.parseInt(parameters.get(CONTENT_DOC_SIZE_FIELD));
    }

    public String getSessionIndex() {
        return parameters.get(SESSION_INDEX_FIELD);
    }

    public String getSessionIdField() {
        return parameters.containsKey(SESSION_ID_FIELD_FIELD)? parameters.get(SESSION_ID_FIELD_FIELD) : "session_id";
    }

    public Integer getSessionSize() {
        if (!parameters.containsKey(SESSION_SIZE_FIELD)) {
            return DEFAULT_SESSION_SIZE;
        }
        return Integer.parseInt(parameters.get(SESSION_SIZE_FIELD));
    }
}
