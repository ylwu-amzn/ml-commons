/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import java.util.List;
import java.util.Map;

import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.commons.authuser.User;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.memorycontainer.MLMemoryContainer;
import org.opensearch.ml.common.memorycontainer.graph.GraphEntity;
import org.opensearch.ml.common.memorycontainer.graph.GraphSearchResult;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.memory.MLGraphSearchAction;
import org.opensearch.ml.common.transport.memory.MLGraphSearchInput;
import org.opensearch.ml.common.transport.memory.MLGraphSearchRequest;
import org.opensearch.ml.common.transport.memory.MLGraphSearchResponse;
import org.opensearch.ml.helper.MemoryContainerHelper;
import org.opensearch.ml.utils.TenantAwareHelper;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

import com.google.common.collect.ImmutableMap;

import lombok.extern.log4j.Log4j2;

/**
 * Transport action for searching graph data within memory containers
 */
@Log4j2
public class TransportGraphSearchAction extends HandledTransportAction<MLGraphSearchRequest, MLGraphSearchResponse> {

    private final MLFeatureEnabledSetting mlFeatureEnabledSetting;
    private final MemoryContainerHelper memoryContainerHelper;
    private final GraphSearchService graphSearchService;

    @Inject
    public TransportGraphSearchAction(
        TransportService transportService,
        ActionFilters actionFilters,
        MLFeatureEnabledSetting mlFeatureEnabledSetting,
        MemoryContainerHelper memoryContainerHelper,
        GraphSearchService graphSearchService
    ) {
        super(MLGraphSearchAction.NAME, transportService, actionFilters, MLGraphSearchRequest::new);
        this.mlFeatureEnabledSetting = mlFeatureEnabledSetting;
        this.memoryContainerHelper = memoryContainerHelper;
        this.graphSearchService = graphSearchService;
    }

    @Override
    protected void doExecute(Task task, MLGraphSearchRequest request, ActionListener<MLGraphSearchResponse> listener) {
        MLGraphSearchInput input = request.getInput();
        String tenantId = request.getTenantId();

        try {
            // Check if agentic memory feature is enabled
            if (!mlFeatureEnabledSetting.isAgenticMemoryEnabled()) {
                throw new MLException("Agentic memory feature is not enabled", RestStatus.FORBIDDEN);
            }

            // Get memory container and validate access
            memoryContainerHelper.getMemoryContainer(input.getMemoryContainerId(), tenantId, ActionListener.wrap(
                container -> {
                    User user = TenantAwareHelper.getUser();

                    // Check if user has access to the container
                    if (!memoryContainerHelper.checkMemoryContainerAccess(container, user)) {
                        listener.onFailure(new MLException(
                            "User does not have access to memory container: " + container.getId(),
                            RestStatus.FORBIDDEN
                        ));
                        return;
                    }

                    // Check if graph is enabled for this container
                    if (container.getConfiguration().getEnableGraph() == null ||
                        !container.getConfiguration().getEnableGraph()) {
                        listener.onFailure(new MLException(
                            "Graph functionality is not enabled for this memory container",
                            RestStatus.BAD_REQUEST
                        ));
                        return;
                    }

                    // Build namespace for search
                    Map<String, String> namespace = buildNamespace(input, user, container);

                    // Execute search based on search type
                    executeGraphSearch(input, container, namespace, user, listener);
                },
                error -> {
                    log.error("Failed to retrieve memory container: {}", input.getMemoryContainerId(), error);
                    listener.onFailure(error);
                }
            ));

        } catch (Exception e) {
            log.error("Graph search failed for container: {}", input.getMemoryContainerId(), e);
            listener.onFailure(e);
        }
    }

    /**
     * Execute the appropriate type of graph search
     */
    private void executeGraphSearch(
        MLGraphSearchInput input,
        MLMemoryContainer container,
        Map<String, String> namespace,
        User user,
        ActionListener<MLGraphSearchResponse> listener
    ) {
        String searchType = input.getSearchType();

        switch (searchType.toLowerCase()) {
            case "text":
                executeTextSearch(input, container, namespace, user, listener);
                break;
            case "traversal":
                executeTraversalSearch(input, container, namespace, user, listener);
                break;
            case "hybrid":
            default:
                executeHybridSearch(input, container, namespace, user, listener);
                break;
        }
    }

    /**
     * Execute text-based entity search
     */
    private void executeTextSearch(
        MLGraphSearchInput input,
        MLMemoryContainer container,
        Map<String, String> namespace,
        User user,
        ActionListener<MLGraphSearchResponse> listener
    ) {
        graphSearchService.searchEntitiesByText(
            input.getQuery(),
            container.getConfiguration(),
            namespace,
            user,
            input.getTopK() != null ? input.getTopK() : 10,
            ActionListener.wrap(
                entities -> {
                    MLGraphSearchResponse response = MLGraphSearchResponse.builder()
                        .entities(entities)
                        .totalCount(entities.size())
                        .query(input.getQuery())
                        .build();
                    listener.onResponse(response);
                },
                error -> {
                    log.error("Text search failed for query: {}", input.getQuery(), error);
                    listener.onFailure(error);
                }
            )
        );
    }

    /**
     * Execute graph traversal from a specific entity
     */
    private void executeTraversalSearch(
        MLGraphSearchInput input,
        MLMemoryContainer container,
        Map<String, String> namespace,
        User user,
        ActionListener<MLGraphSearchResponse> listener
    ) {
        if (input.getEntityId() == null) {
            listener.onFailure(new MLException(
                "Entity ID is required for traversal search",
                RestStatus.BAD_REQUEST
            ));
            return;
        }

        graphSearchService.findRelatedEntities(
            input.getEntityId(),
            container.getConfiguration(),
            namespace,
            user,
            input.getMaxDepth() != null ? input.getMaxDepth() : 2,
            ActionListener.wrap(
                traversalResult -> {
                    MLGraphSearchResponse response = MLGraphSearchResponse.builder()
                        .entities(traversalResult.getEntities())
                        .relationships(traversalResult.getRelationships())
                        .totalCount(traversalResult.getEntities().size())
                        .query(input.getQuery())
                        .build();
                    listener.onResponse(response);
                },
                error -> {
                    log.error("Traversal search failed for entity: {}", input.getEntityId(), error);
                    listener.onFailure(error);
                }
            )
        );
    }

    /**
     * Execute hybrid search combining text similarity and relationship expansion
     */
    private void executeHybridSearch(
        MLGraphSearchInput input,
        MLMemoryContainer container,
        Map<String, String> namespace,
        User user,
        ActionListener<MLGraphSearchResponse> listener
    ) {
        graphSearchService.searchGraphHybrid(
            input.getQuery(),
            container.getConfiguration(),
            namespace,
            user,
            input.getTopK() != null ? input.getTopK() : 10,
            ActionListener.wrap(
                searchResults -> {
                    // Extract entities and relationships from search results
                    List<GraphEntity> entities = searchResults.stream()
                        .map(GraphSearchResult::getEntity)
                        .toList();

                    MLGraphSearchResponse response = MLGraphSearchResponse.builder()
                        .entities(entities)
                        .searchResults(searchResults)
                        .totalCount(entities.size())
                        .query(input.getQuery())
                        .build();
                    listener.onResponse(response);
                },
                error -> {
                    log.error("Hybrid search failed for query: {}", input.getQuery(), error);
                    listener.onFailure(error);
                }
            )
        );
    }

    /**
     * Build namespace map for search operations
     */
    private Map<String, String> buildNamespace(MLGraphSearchInput input, User user, MLMemoryContainer container) {
        ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();

        builder.put("memory_container_id", container.getId());

        if (input.getTenantId() != null) {
            builder.put("tenant_id", input.getTenantId());
        }

        if (user != null && user.getName() != null) {
            builder.put("owner_id", user.getName());
        }

        return builder.build();
    }
}