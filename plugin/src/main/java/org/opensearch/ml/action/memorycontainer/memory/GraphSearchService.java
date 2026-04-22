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
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.SOURCE_ENTITY_FIELD;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.TARGET_ENTITY_FIELD;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.RELATIONSHIP_TYPE_FIELD;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.opensearch.action.search.MultiSearchAction;
import org.opensearch.action.search.MultiSearchRequest;
import org.opensearch.action.search.MultiSearchResponse;
import org.opensearch.action.search.SearchAction;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.commons.authuser.User;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.Strings;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.parameter.MLAlgoParams;
import org.opensearch.ml.common.memorycontainer.MemoryConfiguration;
import org.opensearch.ml.common.memorycontainer.graph.GraphEntity;
import org.opensearch.ml.common.memorycontainer.graph.GraphRelationship;
import org.opensearch.ml.common.memorycontainer.graph.GraphSearchResult;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.engine.algorithms.remote.TextEmbeddingMLRemoteInferenceInput;
import org.opensearch.ml.helper.MemoryContainerHelper;
import org.opensearch.search.SearchHit;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.transport.client.Client;

import com.google.common.annotations.VisibleForTesting;
import com.jayway.jsonpath.JsonPath;

import lombok.extern.log4j.Log4j2;

/**
 * Service for searching and traversing the knowledge graph
 */
@Log4j2
public class GraphSearchService {

    private final Client client;
    private final ClusterService clusterService;
    private final MemoryContainerHelper memoryContainerHelper;

    // Default search parameters
    private static final int DEFAULT_TOP_K = 10;
    private static final int MAX_TRAVERSAL_DEPTH = 5;

    public GraphSearchService(
        Client client,
        ClusterService clusterService,
        MemoryContainerHelper memoryContainerHelper
    ) {
        this.client = client;
        this.clusterService = clusterService;
        this.memoryContainerHelper = memoryContainerHelper;
    }

    /**
     * Search for entities by text query using semantic search
     */
    public void searchEntitiesByText(
        String queryText,
        MemoryConfiguration config,
        Map<String, String> namespace,
        User user,
        int topK,
        ActionListener<List<GraphEntity>> listener
    ) {
        try {
            // Generate query embedding first
            generateQueryEmbedding(queryText, config, ActionListener.wrap(
                embedding -> {
                    // Perform KNN search
                    searchEntitiesByEmbedding(embedding, config, namespace, user, topK, listener);
                },
                error -> {
                    log.error("Failed to generate embedding for query: {}", queryText, error);
                    listener.onFailure(error);
                }
            ));
        } catch (Exception e) {
            log.error("Error in entity text search for query: {}", queryText, e);
            listener.onFailure(e);
        }
    }

    /**
     * Search for entities using pre-computed embedding
     */
    @VisibleForTesting
    void searchEntitiesByEmbedding(
        String embedding,
        MemoryConfiguration config,
        Map<String, String> namespace,
        User user,
        int topK,
        ActionListener<List<GraphEntity>> listener
    ) {
        try {
            String graphNodesIndex = config.getGraphNodesIndexName();

            // Create KNN query
            KNNQueryBuilder knnQuery = new KNNQueryBuilder(
                ENTITY_EMBEDDING_FIELD,
                parseEmbeddingVector(embedding),
                Math.min(topK, 100) // Cap at 100 for performance
            );

            // Apply security and tenant filters
            BoolQueryBuilder boolQuery = QueryBuilders.boolQuery()
                .must(knnQuery);

            // Add container filter
            boolQuery.filter(QueryBuilders.termQuery(MEMORY_CONTAINER_ID_FIELD, namespace.get(MEMORY_CONTAINER_ID_FIELD)));

            // Add tenant isolation
            if (namespace.containsKey(TENANT_ID_FIELD)) {
                boolQuery.filter(QueryBuilders.termQuery(TENANT_ID_FIELD, namespace.get(TENANT_ID_FIELD)));
            }

            // Add user access control if applicable
            if (user != null && !Strings.isNullOrEmpty(user.getName())) {
                boolQuery.filter(QueryBuilders.termQuery(OWNER_ID_FIELD, user.getName()));
            }

            SearchRequest searchRequest = new SearchRequest(graphNodesIndex)
                .source(new SearchSourceBuilder()
                    .query(boolQuery)
                    .size(Math.min(topK, DEFAULT_TOP_K))
                    .fetchSource(true)
                );

            client.execute(SearchAction.INSTANCE, searchRequest, ActionListener.wrap(
                searchResponse -> {
                    List<GraphEntity> entities = parseGraphEntities(searchResponse);
                    listener.onResponse(entities);
                },
                error -> {
                    log.error("Entity search failed for embedding", error);
                    listener.onFailure(error);
                }
            ));
        } catch (Exception e) {
            log.error("Error in entity embedding search", e);
            listener.onFailure(e);
        }
    }

    /**
     * Find entities related to a given entity through relationships
     */
    public void findRelatedEntities(
        String entityId,
        MemoryConfiguration config,
        Map<String, String> namespace,
        User user,
        int maxDepth,
        ActionListener<GraphTraversalResult> listener
    ) {
        try {
            int safeDepth = Math.min(maxDepth, MAX_TRAVERSAL_DEPTH);
            Set<String> visitedEntities = new HashSet<>();
            Map<String, GraphEntity> entityMap = new HashMap<>();
            List<GraphRelationship> relationships = new ArrayList<>();

            // Start traversal from the given entity
            traverseGraph(
                List.of(entityId),
                safeDepth,
                config,
                namespace,
                user,
                visitedEntities,
                entityMap,
                relationships,
                ActionListener.wrap(
                    success -> {
                        GraphTraversalResult result = GraphTraversalResult.builder()
                            .entities(new ArrayList<>(entityMap.values()))
                            .relationships(relationships)
                            .centerEntityId(entityId)
                            .traversalDepth(safeDepth)
                            .build();
                        listener.onResponse(result);
                    },
                    listener::onFailure
                )
            );
        } catch (Exception e) {
            log.error("Error in graph traversal for entity: {}", entityId, e);
            listener.onFailure(e);
        }
    }

    /**
     * Recursive graph traversal implementation
     */
    @VisibleForTesting
    void traverseGraph(
        List<String> currentEntityIds,
        int remainingDepth,
        MemoryConfiguration config,
        Map<String, String> namespace,
        User user,
        Set<String> visitedEntities,
        Map<String, GraphEntity> entityMap,
        List<GraphRelationship> relationships,
        ActionListener<Void> listener
    ) {
        if (remainingDepth <= 0 || currentEntityIds.isEmpty()) {
            listener.onResponse(null);
            return;
        }

        // Find relationships for current entities
        findRelationshipsForEntities(
            currentEntityIds,
            config,
            namespace,
            user,
            ActionListener.wrap(
                foundRelationships -> {
                    relationships.addAll(foundRelationships);

                    // Collect next level entity IDs
                    Set<String> nextEntityIds = new HashSet<>();
                    for (GraphRelationship rel : foundRelationships) {
                        String sourceId = rel.getSourceEntityId();
                        String targetId = rel.getTargetEntityId();

                        // Add unvisited entities to next level
                        if (!visitedEntities.contains(sourceId)) {
                            nextEntityIds.add(sourceId);
                        }
                        if (!visitedEntities.contains(targetId)) {
                            nextEntityIds.add(targetId);
                        }
                    }

                    // Mark current entities as visited
                    visitedEntities.addAll(currentEntityIds);

                    // Fetch entity details for new entities
                    if (!nextEntityIds.isEmpty()) {
                        fetchEntityDetails(
                            nextEntityIds,
                            config,
                            namespace,
                            user,
                            ActionListener.wrap(
                                fetchedEntities -> {
                                    // Add entities to map
                                    for (GraphEntity entity : fetchedEntities) {
                                        entityMap.put(entity.getEntityId(), entity);
                                    }

                                    // Continue traversal with next level
                                    traverseGraph(
                                        new ArrayList<>(nextEntityIds),
                                        remainingDepth - 1,
                                        config,
                                        namespace,
                                        user,
                                        visitedEntities,
                                        entityMap,
                                        relationships,
                                        listener
                                    );
                                },
                                error -> {
                                    log.warn("Failed to fetch entity details during traversal", error);
                                    // Continue without entity details
                                    traverseGraph(
                                        new ArrayList<>(nextEntityIds),
                                        remainingDepth - 1,
                                        config,
                                        namespace,
                                        user,
                                        visitedEntities,
                                        entityMap,
                                        relationships,
                                        listener
                                    );
                                }
                            )
                        );
                    } else {
                        listener.onResponse(null);
                    }
                },
                error -> {
                    log.error("Failed to find relationships for entities: {}", currentEntityIds, error);
                    listener.onFailure(error);
                }
            )
        );
    }

    /**
     * Find relationships involving the given entity IDs
     */
    @VisibleForTesting
    void findRelationshipsForEntities(
        List<String> entityIds,
        MemoryConfiguration config,
        Map<String, String> namespace,
        User user,
        ActionListener<List<GraphRelationship>> listener
    ) {
        try {
            String graphEdgesIndex = config.getGraphEdgesIndexName();

            // Create query for relationships where entities are source or target
            BoolQueryBuilder relationshipQuery = QueryBuilders.boolQuery();

            BoolQueryBuilder entityFilter = QueryBuilders.boolQuery();
            for (String entityId : entityIds) {
                entityFilter.should(QueryBuilders.termQuery(SOURCE_ENTITY_FIELD, entityId));
                entityFilter.should(QueryBuilders.termQuery(TARGET_ENTITY_FIELD, entityId));
            }
            entityFilter.minimumShouldMatch(1);

            relationshipQuery.must(entityFilter);

            // Add container and tenant filters
            relationshipQuery.filter(QueryBuilders.termQuery(MEMORY_CONTAINER_ID_FIELD, namespace.get(MEMORY_CONTAINER_ID_FIELD)));
            if (namespace.containsKey(TENANT_ID_FIELD)) {
                relationshipQuery.filter(QueryBuilders.termQuery(TENANT_ID_FIELD, namespace.get(TENANT_ID_FIELD)));
            }

            // Add user access control
            if (user != null && !Strings.isNullOrEmpty(user.getName())) {
                relationshipQuery.filter(QueryBuilders.termQuery(OWNER_ID_FIELD, user.getName()));
            }

            // Only active relationships
            relationshipQuery.filter(QueryBuilders.termQuery("is_active", true));

            SearchRequest searchRequest = new SearchRequest(graphEdgesIndex)
                .source(new SearchSourceBuilder()
                    .query(relationshipQuery)
                    .size(1000) // Large limit for relationship discovery
                    .fetchSource(true)
                );

            client.execute(SearchAction.INSTANCE, searchRequest, ActionListener.wrap(
                searchResponse -> {
                    List<GraphRelationship> relationships = parseGraphRelationships(searchResponse);
                    listener.onResponse(relationships);
                },
                error -> {
                    log.error("Relationship search failed for entities: {}", entityIds, error);
                    listener.onFailure(error);
                }
            ));
        } catch (Exception e) {
            log.error("Error searching relationships for entities: {}", entityIds, e);
            listener.onFailure(e);
        }
    }

    /**
     * Fetch detailed entity information for the given entity IDs
     */
    @VisibleForTesting
    void fetchEntityDetails(
        Set<String> entityIds,
        MemoryConfiguration config,
        Map<String, String> namespace,
        User user,
        ActionListener<List<GraphEntity>> listener
    ) {
        try {
            String graphNodesIndex = config.getGraphNodesIndexName();

            // Create query for specific entity IDs
            BoolQueryBuilder entityQuery = QueryBuilders.boolQuery()
                .must(QueryBuilders.idsQuery().addIds(entityIds.toArray(new String[0])));

            // Add filters
            entityQuery.filter(QueryBuilders.termQuery(MEMORY_CONTAINER_ID_FIELD, namespace.get(MEMORY_CONTAINER_ID_FIELD)));
            if (namespace.containsKey(TENANT_ID_FIELD)) {
                entityQuery.filter(QueryBuilders.termQuery(TENANT_ID_FIELD, namespace.get(TENANT_ID_FIELD)));
            }

            SearchRequest searchRequest = new SearchRequest(graphNodesIndex)
                .source(new SearchSourceBuilder()
                    .query(entityQuery)
                    .size(entityIds.size())
                    .fetchSource(true)
                );

            client.execute(SearchAction.INSTANCE, searchRequest, ActionListener.wrap(
                searchResponse -> {
                    List<GraphEntity> entities = parseGraphEntities(searchResponse);
                    listener.onResponse(entities);
                },
                error -> {
                    log.error("Entity detail fetch failed for IDs: {}", entityIds, error);
                    listener.onFailure(error);
                }
            ));
        } catch (Exception e) {
            log.error("Error fetching entity details for IDs: {}", entityIds, e);
            listener.onFailure(e);
        }
    }

    /**
     * Hybrid graph search using both semantic similarity and relationship traversal
     */
    public void searchGraphHybrid(
        String queryText,
        MemoryConfiguration config,
        Map<String, String> namespace,
        User user,
        int topK,
        ActionListener<List<GraphSearchResult>> listener
    ) {
        try {
            // First, find entities by text similarity
            searchEntitiesByText(queryText, config, namespace, user, topK / 2, ActionListener.wrap(
                textEntities -> {
                    if (textEntities.isEmpty()) {
                        listener.onResponse(new ArrayList<>());
                        return;
                    }

                    // Then, expand with related entities
                    List<String> seedEntityIds = textEntities.stream()
                        .limit(3) // Use top 3 as seeds to avoid explosion
                        .map(GraphEntity::getEntityId)
                        .toList();

                    findRelatedEntitiesMultiple(
                        seedEntityIds,
                        config,
                        namespace,
                        user,
                        2, // Depth 2 for hybrid search
                        ActionListener.wrap(
                            relatedResults -> {
                                List<GraphSearchResult> hybridResults = combineSearchResults(
                                    textEntities,
                                    relatedResults,
                                    queryText,
                                    topK
                                );
                                listener.onResponse(hybridResults);
                            },
                            error -> {
                                log.warn("Related entity search failed in hybrid search, returning text results only", error);
                                // Fall back to text-only results
                                List<GraphSearchResult> textResults = textEntities.stream()
                                    .map(entity -> GraphSearchResult.builder()
                                        .entity(entity)
                                        .score(1.0f) // Default score for text match
                                        .matchType("text_similarity")
                                        .build())
                                    .toList();
                                listener.onResponse(textResults);
                            }
                        )
                    );
                },
                error -> {
                    log.error("Text search failed in hybrid search for query: {}", queryText, error);
                    listener.onFailure(error);
                }
            ));
        } catch (Exception e) {
            log.error("Error in hybrid graph search for query: {}", queryText, e);
            listener.onFailure(e);
        }
    }

    /**
     * Find related entities for multiple seed entities
     */
    @VisibleForTesting
    void findRelatedEntitiesMultiple(
        List<String> seedEntityIds,
        MemoryConfiguration config,
        Map<String, String> namespace,
        User user,
        int maxDepth,
        ActionListener<List<GraphTraversalResult>> listener
    ) {
        List<GraphTraversalResult> results = new ArrayList<>();

        // Create listeners for parallel traversals
        ActionListener<GraphTraversalResult> collectingListener = ActionListener.wrap(
            result -> {
                synchronized (results) {
                    results.add(result);
                    if (results.size() == seedEntityIds.size()) {
                        listener.onResponse(results);
                    }
                }
            },
            error -> {
                log.warn("Single entity traversal failed during multi-traversal", error);
                synchronized (results) {
                    // Add empty result to maintain count
                    results.add(GraphTraversalResult.builder()
                        .entities(new ArrayList<>())
                        .relationships(new ArrayList<>())
                        .build());
                    if (results.size() == seedEntityIds.size()) {
                        listener.onResponse(results);
                    }
                }
            }
        );

        // Start parallel traversals
        for (String entityId : seedEntityIds) {
            findRelatedEntities(entityId, config, namespace, user, maxDepth, collectingListener);
        }
    }

    /**
     * Combine text search and graph traversal results
     */
    @VisibleForTesting
    List<GraphSearchResult> combineSearchResults(
        List<GraphEntity> textEntities,
        List<GraphTraversalResult> traversalResults,
        String queryText,
        int topK
    ) {
        Map<String, GraphSearchResult> resultMap = new HashMap<>();

        // Add text similarity results (higher base score)
        for (GraphEntity entity : textEntities) {
            GraphSearchResult result = GraphSearchResult.builder()
                .entity(entity)
                .score(1.0f)
                .matchType("text_similarity")
                .queryText(queryText)
                .build();
            resultMap.put(entity.getEntityId(), result);
        }

        // Add related entities (lower base score, boosted by relationship count)
        for (GraphTraversalResult traversal : traversalResults) {
            for (GraphEntity entity : traversal.getEntities()) {
                if (!resultMap.containsKey(entity.getEntityId())) {
                    // Count incoming relationships for this entity
                    long relationshipCount = traversal.getRelationships().stream()
                        .filter(rel -> rel.getTargetEntityId().equals(entity.getEntityId()) ||
                                      rel.getSourceEntityId().equals(entity.getEntityId()))
                        .count();

                    float score = 0.5f + (relationshipCount * 0.1f); // Base + relationship bonus

                    GraphSearchResult result = GraphSearchResult.builder()
                        .entity(entity)
                        .score(score)
                        .matchType("relationship_expansion")
                        .relationshipCount((int) relationshipCount)
                        .queryText(queryText)
                        .build();
                    resultMap.put(entity.getEntityId(), result);
                }
            }
        }

        // Sort by score and return top K
        return resultMap.values().stream()
            .sorted((a, b) -> Float.compare(b.getScore(), a.getScore()))
            .limit(topK)
            .toList();
    }

    // Helper methods for parsing and embedding generation

    /**
     * Generate embedding for search query
     */
    @VisibleForTesting
    void generateQueryEmbedding(
        String queryText,
        MemoryConfiguration config,
        ActionListener<String> listener
    ) {
        try {
            TextEmbeddingMLRemoteInferenceInput embeddingInput = TextEmbeddingMLRemoteInferenceInput.builder()
                .inputText(List.of(queryText))
                .build();

            MLInput mlInput = MLInput.builder()
                .algorithm(MLAlgoParams.TEXT_EMBEDDING)
                .inputDataset(embeddingInput)
                .build();

            MLPredictionTaskRequest embeddingRequest = new MLPredictionTaskRequest(
                config.getEmbeddingModelId(),
                mlInput,
                null
            );

            client.execute(MLPredictionTaskAction.INSTANCE, embeddingRequest, ActionListener.wrap(
                response -> {
                    try {
                        String embeddingJson = parseEmbeddingFromResponse(response);
                        listener.onResponse(embeddingJson);
                    } catch (Exception e) {
                        log.error("Failed to parse embedding response for query: {}", queryText, e);
                        listener.onFailure(e);
                    }
                },
                error -> {
                    log.error("Embedding generation failed for query: {}", queryText, error);
                    listener.onFailure(error);
                }
            ));
        } catch (Exception e) {
            log.error("Error creating embedding request for query: {}", queryText, e);
            listener.onFailure(e);
        }
    }

    /**
     * Parse embedding from ML response
     */
    private String parseEmbeddingFromResponse(MLTaskResponse response) throws IOException {
        String responseJson = response.getOutput().toString();
        return JsonPath.read(responseJson, "$.inference_results[0].output[0].data");
    }

    /**
     * Parse embedding vector from JSON string
     */
    private float[] parseEmbeddingVector(String embeddingJson) throws IOException {
        List<Float> values = JsonPath.read(embeddingJson, "$");
        float[] array = new float[values.size()];
        for (int i = 0; i < values.size(); i++) {
            array[i] = values.get(i);
        }
        return array;
    }

    /**
     * Parse graph entities from search response
     */
    private List<GraphEntity> parseGraphEntities(SearchResponse searchResponse) {
        return List.of(searchResponse.getHits().getHits()).stream()
            .map(this::mapToGraphEntity)
            .toList();
    }

    /**
     * Parse graph relationships from search response
     */
    private List<GraphRelationship> parseGraphRelationships(SearchResponse searchResponse) {
        return List.of(searchResponse.getHits().getHits()).stream()
            .map(this::mapToGraphRelationship)
            .toList();
    }

    /**
     * Map search hit to GraphEntity
     */
    private GraphEntity mapToGraphEntity(SearchHit hit) {
        Map<String, Object> source = hit.getSourceAsMap();
        return GraphEntity.builder()
            .entityId(hit.getId())
            .name((String) source.get(ENTITY_NAME_FIELD))
            .type((String) source.get(ENTITY_TYPE_FIELD))
            .confidence((Double) source.getOrDefault("confidence", 0.0))
            .mentionCount((Integer) source.getOrDefault("mention_count", 1))
            .build();
    }

    /**
     * Map search hit to GraphRelationship
     */
    private GraphRelationship mapToGraphRelationship(SearchHit hit) {
        Map<String, Object> source = hit.getSourceAsMap();
        return GraphRelationship.builder()
            .relationshipId(hit.getId())
            .sourceEntityId((String) source.get(SOURCE_ENTITY_FIELD))
            .targetEntityId((String) source.get(TARGET_ENTITY_FIELD))
            .relationshipType((String) source.get(RELATIONSHIP_TYPE_FIELD))
            .confidence((Double) source.getOrDefault("confidence", 0.0))
            .build();
    }
}