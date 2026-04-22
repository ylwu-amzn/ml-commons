/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import static org.opensearch.ml.common.CommonValue.TENANT_ID_FIELD;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.MEMORY_CONTAINER_ID_FIELD;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.OWNER_ID_FIELD;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.ENTITY_ID_FIELD;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.ENTITY_NAME_FIELD;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.ENTITY_TYPE_FIELD;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.ENTITY_EMBEDDING_FIELD;
import static org.opensearch.common.xcontent.json.JsonXContent.jsonXContent;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.codec.digest.DigestUtils;
import org.opensearch.action.search.SearchAction;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.memorycontainer.MemoryConfiguration;
import org.opensearch.ml.common.memorycontainer.MemoryType;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.search.SearchHit;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.transport.client.Client;

import com.google.common.annotations.VisibleForTesting;
import com.jayway.jsonpath.JsonPath;

import lombok.extern.log4j.Log4j2;

/**
 * Service for entity deduplication using similarity search and LLM verification
 */
@Log4j2
public class EntityDeduplicationService {

    private final Client client;

    // Similarity thresholds
    private static final double AUTO_MERGE_THRESHOLD = 0.8;
    private static final double AUTO_NEW_THRESHOLD = 0.6;

    // LLM verification prompt template
    private static final String ENTITY_VERIFICATION_PROMPT = """
        Compare these two entities and determine if they refer to the same real-world entity:

        Entity 1: "%s" (type: %s)
        Entity 2: "%s" (type: %s)

        Consider variations in naming, abbreviations, and aliases.
        Respond with JSON: {"same_entity": true/false, "confidence": 0.0-1.0}
        """;

    public EntityDeduplicationService(Client client) {
        this.client = client;
    }

    /**
     * Deduplicate an extracted entity against existing entities.
     * Uses neural search to delegate embedding to the neural-search plugin.
     * Gracefully falls back to creating a new entity on search failure.
     */
    public void deduplicateEntity(
        ExtractedEntity entity,
        MemoryConfiguration config,
        Map<String, String> namespace,
        ActionListener<ProcessedEntity> listener
    ) {
        try {
            String entityText = entity.getName() + " [" + entity.getType() + "]";
            searchSimilarEntities(entity, entityText, config, namespace, ActionListener.wrap(
                similarEntities -> processSimilarityResults(entity, similarEntities, config, namespace, listener),
                error -> {
                    log.error("Failed to search similar entities for entity: {}", entity.getName(), error);
                    ProcessedEntity newEntity = createNewEntity(entity, config, namespace);
                    listener.onResponse(newEntity);
                }
            ));
        } catch (Exception e) {
            log.error("Error in entity deduplication for entity: {}", entity.getName(), e);
            ProcessedEntity newEntity = createNewEntity(entity, config, namespace);
            listener.onResponse(newEntity);
        }
    }

    @VisibleForTesting
    void searchSimilarEntities(
        ExtractedEntity entity,
        String entityText,
        MemoryConfiguration config,
        Map<String, String> namespace,
        ActionListener<List<SimilarEntity>> listener
    ) {
        try {
            String graphNodesIndex = config.getGraphNodesIndexName();

            if (config.getEmbeddingModelType() != FunctionName.TEXT_EMBEDDING
                && config.getEmbeddingModelType() != FunctionName.SPARSE_ENCODING) {
                listener.onFailure(
                    new IllegalStateException("Unsupported embedding model type for dedup: " + config.getEmbeddingModelType())
                );
                return;
            }
            String neuralType = config.getEmbeddingModelType() == FunctionName.TEXT_EMBEDDING ? "neural" : "neural_sparse";

            String neuralQuery;
            try (XContentBuilder builder = XContentBuilder.builder(jsonXContent)) {
                neuralQuery = builder
                    .startObject()
                    .startObject(neuralType)
                    .startObject(ENTITY_EMBEDDING_FIELD)
                    .field("query_text", entityText)
                    .field("model_id", config.getEmbeddingModelId())
                    .field("k", 10)
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
            }

            BoolQueryBuilder boolQuery = QueryBuilders.boolQuery()
                .must(QueryBuilders.wrapperQuery(neuralQuery))
                .filter(QueryBuilders.termQuery(MEMORY_CONTAINER_ID_FIELD, namespace.get(MEMORY_CONTAINER_ID_FIELD)))
                .filter(QueryBuilders.termQuery(ENTITY_TYPE_FIELD, entity.getType()));

            if (namespace.containsKey(TENANT_ID_FIELD)) {
                boolQuery.filter(QueryBuilders.termQuery(TENANT_ID_FIELD, namespace.get(TENANT_ID_FIELD)));
            }

            SearchRequest searchRequest = new SearchRequest(graphNodesIndex)
                .source(new SearchSourceBuilder().query(boolQuery).size(10));

            client.execute(SearchAction.INSTANCE, searchRequest, ActionListener.wrap(
                searchResponse -> listener.onResponse(parseSimilarEntities(searchResponse)),
                error -> {
                    log.error("Similar-entity search failed for entity: {}", entity.getName(), error);
                    listener.onFailure(error);
                }
            ));
        } catch (IOException e) {
            log.error("Failed to build neural query for entity: {}", entity.getName(), e);
            listener.onFailure(e);
        } catch (Exception e) {
            log.error("Error in similar-entity search for entity: {}", entity.getName(), e);
            listener.onFailure(e);
        }
    }

    /**
     * Process similarity results and make deduplication decision
     */
    @VisibleForTesting
    void processSimilarityResults(
        ExtractedEntity entity,
        List<SimilarEntity> similarEntities,
        MemoryConfiguration config,
        Map<String, String> namespace,
        ActionListener<ProcessedEntity> listener
    ) {
        if (similarEntities.isEmpty()) {
            // No similar entities found - create new
            ProcessedEntity newEntity = createNewEntity(entity, config, namespace);
            listener.onResponse(newEntity);
            return;
        }

        SimilarEntity mostSimilar = similarEntities.get(0);
        double similarity = mostSimilar.getScore();

        if (similarity >= AUTO_MERGE_THRESHOLD) {
            // High similarity - auto merge
            ProcessedEntity mergedEntity = createMergedEntity(entity, mostSimilar, config, namespace);
            listener.onResponse(mergedEntity);
        } else if (similarity <= AUTO_NEW_THRESHOLD) {
            // Low similarity - create new
            ProcessedEntity newEntity = createNewEntity(entity, config, namespace);
            listener.onResponse(newEntity);
        } else {
            // Ambiguous similarity - use LLM verification
            verifyEntitySimilarityWithLLM(entity, mostSimilar, config, ActionListener.wrap(
                isSameEntity -> {
                    ProcessedEntity result = isSameEntity
                        ? createMergedEntity(entity, mostSimilar, config, namespace)
                        : createNewEntity(entity, config, namespace);
                    listener.onResponse(result);
                },
                error -> {
                    log.warn("LLM verification failed for entity: {}, creating new entity", entity.getName(), error);
                    ProcessedEntity newEntity = createNewEntity(entity, config, namespace);
                    listener.onResponse(newEntity);
                }
            ));
        }
    }

    /**
     * Verify entity similarity using LLM when similarity is ambiguous
     */
    @VisibleForTesting
    void verifyEntitySimilarityWithLLM(
        ExtractedEntity entity,
        SimilarEntity similar,
        MemoryConfiguration config,
        ActionListener<Boolean> listener
    ) {
        try {
            String prompt = String.format(
                ENTITY_VERIFICATION_PROMPT,
                entity.getName(),
                entity.getType(),
                similar.getName(),
                similar.getType()
            );

            Map<String, String> parameters = new HashMap<>();
            parameters.put("user_prompt", prompt);

            RemoteInferenceInputDataSet inputDataSet = RemoteInferenceInputDataSet.builder()
                .parameters(parameters)
                .build();

            MLInput mlInput = MLInput.builder()
                .algorithm(FunctionName.REMOTE)
                .inputDataset(inputDataSet)
                .build();

            MLPredictionTaskRequest request = new MLPredictionTaskRequest(
                config.getLlmId(),
                mlInput
            );

            client.execute(MLPredictionTaskAction.INSTANCE, request, ActionListener.wrap(
                response -> {
                    try {
                        boolean isSame = parseVerificationResponse(response);
                        listener.onResponse(isSame);
                    } catch (Exception e) {
                        log.error("Failed to parse LLM verification response", e);
                        listener.onFailure(e);
                    }
                },
                error -> {
                    log.error("LLM verification request failed", error);
                    listener.onFailure(error);
                }
            ));
        } catch (Exception e) {
            log.error("Error creating LLM verification request", e);
            listener.onFailure(e);
        }
    }

    /**
     * Create a new entity with generated ID
     */
    @VisibleForTesting
    ProcessedEntity createNewEntity(
        ExtractedEntity entity,
        MemoryConfiguration config,
        Map<String, String> namespace
    ) {
        String entityId = generateEntityId(entity.getName(), config, namespace);

        return ProcessedEntity.builder()
            .entityId(entityId)
            .name(entity.getName())
            .type(entity.getType())
            .confidence(entity.getConfidence())
            .isNewEntity(true)
            .existingEntityId(null)
            .mentionCount(1)
            .build();
    }

    /**
     * Create a merged entity referencing existing entity
     */
    @VisibleForTesting
    ProcessedEntity createMergedEntity(
        ExtractedEntity entity,
        SimilarEntity similar,
        MemoryConfiguration config,
        Map<String, String> namespace
    ) {
        return ProcessedEntity.builder()
            .entityId(similar.getEntityId())
            .name(entity.getName()) // Keep new mention name
            .type(entity.getType())
            .confidence(Math.max(entity.getConfidence(), similar.getConfidence()))
            .isNewEntity(false)
            .existingEntityId(similar.getEntityId())
            .mentionCount(similar.getMentionCount() + 1)
            .build();
    }

    /**
     * Generate unique entity ID using configurable scope
     */
    @VisibleForTesting
    String generateEntityId(
        String entityName,
        MemoryConfiguration config,
        Map<String, String> namespace
    ) {
        String scope = extractScopeValue(config.getEntityScope(), namespace);
        String tenantId = namespace.get(TENANT_ID_FIELD);
        String normalizedName = entityName.toLowerCase().trim().replaceAll("\\s+", " ");

        // Use format: tenant_id + \0 + scope + \0 + entity_name
        String input = tenantId + "\0" + scope + "\0" + normalizedName;
        return "mn:" + DigestUtils.sha256Hex(input);
    }

    /**
     * Extract scope value based on configuration
     */
    @VisibleForTesting
    String extractScopeValue(String entityScope, Map<String, String> namespace) {
        if ("user_id".equals(entityScope)) {
            return namespace.get(OWNER_ID_FIELD);
        } else if ("container_id".equals(entityScope)) {
            return namespace.get(MEMORY_CONTAINER_ID_FIELD);
        } else if ("tenant_id".equals(entityScope)) {
            return namespace.get(TENANT_ID_FIELD);
        } else {
            log.warn("Unknown entity scope: {}, defaulting to user_id", entityScope);
            return namespace.get(OWNER_ID_FIELD);
        }
    }

    /**
     * Parse similar entities from search response
     */
    private List<SimilarEntity> parseSimilarEntities(SearchResponse searchResponse) {
        return List.of(searchResponse.getHits().getHits()).stream()
            .map(this::mapToSimilarEntity)
            .toList();
    }

    /**
     * Map search hit to SimilarEntity
     */
    private SimilarEntity mapToSimilarEntity(SearchHit hit) {
        Map<String, Object> source = hit.getSourceAsMap();
        return SimilarEntity.builder()
            .entityId(hit.getId())
            .name((String) source.get(ENTITY_NAME_FIELD))
            .type((String) source.get(ENTITY_TYPE_FIELD))
            .confidence((Double) source.getOrDefault("confidence", 0.0))
            .score(hit.getScore())
            .mentionCount((Integer) source.getOrDefault("mention_count", 1))
            .build();
    }

    /**
     * Parse LLM verification response
     */
    private boolean parseVerificationResponse(MLTaskResponse response) throws IOException {
        String responseJson = response.getOutput().toString();
        try {
            return JsonPath.read(responseJson, "$.same_entity");
        } catch (Exception e) {
            log.warn("Failed to parse structured LLM response, falling back to text analysis: {}", responseJson);
            // Fallback: simple text analysis for boolean response
            String lowercaseResponse = responseJson.toLowerCase();
            return lowercaseResponse.contains("true") || lowercaseResponse.contains("\"same_entity\":true");
        }
    }

    /**
     * Represents a similar entity found during deduplication search
     */
    public static class SimilarEntity {
        private final String entityId;
        private final String name;
        private final String type;
        private final double confidence;
        private final float score;
        private final int mentionCount;

        private SimilarEntity(Builder builder) {
            this.entityId = builder.entityId;
            this.name = builder.name;
            this.type = builder.type;
            this.confidence = builder.confidence;
            this.score = builder.score;
            this.mentionCount = builder.mentionCount;
        }

        public static Builder builder() {
            return new Builder();
        }

        public String getEntityId() { return entityId; }
        public String getName() { return name; }
        public String getType() { return type; }
        public double getConfidence() { return confidence; }
        public float getScore() { return score; }
        public int getMentionCount() { return mentionCount; }

        public static class Builder {
            private String entityId;
            private String name;
            private String type;
            private double confidence;
            private float score;
            private int mentionCount = 1;

            public Builder entityId(String entityId) {
                this.entityId = entityId;
                return this;
            }

            public Builder name(String name) {
                this.name = name;
                return this;
            }

            public Builder type(String type) {
                this.type = type;
                return this;
            }

            public Builder confidence(double confidence) {
                this.confidence = confidence;
                return this;
            }

            public Builder score(float score) {
                this.score = score;
                return this;
            }

            public Builder mentionCount(int mentionCount) {
                this.mentionCount = mentionCount;
                return this;
            }

            public SimilarEntity build() {
                return new SimilarEntity(this);
            }
        }
    }
}