/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.ml.common.memorycontainer.MLMemoryContainer;
import org.opensearch.ml.common.memorycontainer.MemoryConfiguration;
import org.opensearch.ml.common.memorycontainer.graph.GraphEntity;
import org.opensearch.ml.common.memorycontainer.graph.GraphRelationship;
import org.opensearch.ml.common.memorycontainer.graph.GraphSearchResult;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.memory.MLGraphSearchInput;
import org.opensearch.ml.common.transport.memory.MLGraphSearchRequest;
import org.opensearch.ml.common.transport.memory.MLGraphSearchResponse;
import org.opensearch.ml.helper.MemoryContainerHelper;
import org.opensearch.tasks.Task;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;
import org.opensearch.transport.client.Client;

public class TransportGraphSearchActionTests extends OpenSearchTestCase {

    @Mock
    private TransportService transportService;

    @Mock
    private ActionFilters actionFilters;

    @Mock
    private MLFeatureEnabledSetting mlFeatureEnabledSetting;

    @Mock
    private MemoryContainerHelper memoryContainerHelper;

    @Mock
    private GraphSearchService graphSearchService;

    @Mock
    private Client client;

    @Mock
    private ThreadPool threadPool;

    @Mock
    private Task task;

    @Mock
    private ActionListener<MLGraphSearchResponse> actionListener;

    private ThreadContext threadContext;
    private TransportGraphSearchAction action;
    private MLMemoryContainer graphEnabledContainer;
    private MLMemoryContainer graphDisabledContainer;

    private static final String CONTAINER_ID = "container-123";
    private static final String TENANT_ID = "tenant-1";
    private static final String QUERY = "who knows alice";

    @Before
    public void setUp() throws Exception {
        super.setUp();
        MockitoAnnotations.openMocks(this);

        threadContext = new ThreadContext(Settings.EMPTY);
        when(client.threadPool()).thenReturn(threadPool);
        when(threadPool.getThreadContext()).thenReturn(threadContext);

        when(mlFeatureEnabledSetting.isAgenticMemoryEnabled()).thenReturn(true);

        graphEnabledContainer = MLMemoryContainer.builder()
            .name("graph-container")
            .configuration(
                MemoryConfiguration.builder()
                    .indexPrefix("test-memory")
                    .enableGraph(true)
                    .build()
            )
            .build();

        graphDisabledContainer = MLMemoryContainer.builder()
            .name("plain-container")
            .configuration(MemoryConfiguration.builder().indexPrefix("test-memory").build())
            .build();

        action = new TransportGraphSearchAction(
            transportService,
            actionFilters,
            mlFeatureEnabledSetting,
            memoryContainerHelper,
            graphSearchService,
            client
        );
    }

    private MLGraphSearchRequest request(MLGraphSearchInput input) {
        return MLGraphSearchRequest.builder().input(input).tenantId(TENANT_ID).build();
    }

    private MLGraphSearchInput baseInput(String searchType) {
        return MLGraphSearchInput.builder()
            .memoryContainerId(CONTAINER_ID)
            .query(QUERY)
            .searchType(searchType)
            .topK(5)
            .maxDepth(3)
            .build();
    }

    private void mockGetContainer(MLMemoryContainer container) {
        doAnswer(invocation -> {
            ActionListener<MLMemoryContainer> listener = invocation.getArgument(2);
            listener.onResponse(container);
            return null;
        }).when(memoryContainerHelper).getMemoryContainer(eq(CONTAINER_ID), eq(TENANT_ID), any());
    }

    @Test
    public void testDoExecute_FeatureDisabled() {
        when(mlFeatureEnabledSetting.isAgenticMemoryEnabled()).thenReturn(false);

        action.doExecute(task, request(baseInput("hybrid")), actionListener);

        ArgumentCaptor<Exception> captor = ArgumentCaptor.forClass(Exception.class);
        verify(actionListener).onFailure(captor.capture());
        verify(memoryContainerHelper, never()).getMemoryContainer(any(), any(), any());
        assertTrue(captor.getValue() instanceof OpenSearchStatusException);
        assertEquals(RestStatus.FORBIDDEN, ((OpenSearchStatusException) captor.getValue()).status());
    }

    @Test
    public void testDoExecute_GetContainerFailure() {
        Exception expected = new RuntimeException("container not found");
        doAnswer(invocation -> {
            ActionListener<MLMemoryContainer> listener = invocation.getArgument(2);
            listener.onFailure(expected);
            return null;
        }).when(memoryContainerHelper).getMemoryContainer(eq(CONTAINER_ID), eq(TENANT_ID), any());

        action.doExecute(task, request(baseInput("hybrid")), actionListener);

        verify(actionListener).onFailure(expected);
        verify(graphSearchService, never()).searchEntitiesByText(any(), any(), any(), any(), anyInt(), any());
        verify(graphSearchService, never()).searchGraphHybrid(any(), any(), any(), any(), anyInt(), any());
        verify(graphSearchService, never()).findRelatedEntities(any(), any(), any(), any(), anyInt(), any());
    }

    @Test
    public void testDoExecute_AccessDenied() {
        mockGetContainer(graphEnabledContainer);
        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphEnabledContainer))).thenReturn(false);

        action.doExecute(task, request(baseInput("hybrid")), actionListener);

        ArgumentCaptor<Exception> captor = ArgumentCaptor.forClass(Exception.class);
        verify(actionListener).onFailure(captor.capture());
        assertTrue(captor.getValue() instanceof OpenSearchStatusException);
        assertEquals(RestStatus.FORBIDDEN, ((OpenSearchStatusException) captor.getValue()).status());
        verify(graphSearchService, never()).searchGraphHybrid(any(), any(), any(), any(), anyInt(), any());
    }

    @Test
    public void testDoExecute_GraphDisabled() {
        mockGetContainer(graphDisabledContainer);
        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphDisabledContainer))).thenReturn(true);

        action.doExecute(task, request(baseInput("hybrid")), actionListener);

        ArgumentCaptor<Exception> captor = ArgumentCaptor.forClass(Exception.class);
        verify(actionListener).onFailure(captor.capture());
        assertTrue(captor.getValue() instanceof OpenSearchStatusException);
        assertEquals(RestStatus.BAD_REQUEST, ((OpenSearchStatusException) captor.getValue()).status());
        assertTrue(captor.getValue().getMessage().contains("Graph functionality is not enabled"));
    }

    @Test
    public void testDoExecute_TextSearch_Success() {
        mockGetContainer(graphEnabledContainer);
        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphEnabledContainer))).thenReturn(true);

        GraphEntity entity = GraphEntity.builder().entityId("e1").name("Alice").type("person").build();
        doAnswer(invocation -> {
            ActionListener<List<GraphEntity>> listener = invocation.getArgument(5);
            listener.onResponse(Arrays.asList(entity));
            return null;
        }).when(graphSearchService).searchEntitiesByText(eq(QUERY), any(), any(), any(), eq(5), any());

        action.doExecute(task, request(baseInput("text")), actionListener);

        ArgumentCaptor<MLGraphSearchResponse> captor = ArgumentCaptor.forClass(MLGraphSearchResponse.class);
        verify(actionListener, times(1)).onResponse(captor.capture());
        verify(actionListener, never()).onFailure(any());
        verify(graphSearchService, never()).searchGraphHybrid(any(), any(), any(), any(), anyInt(), any());
        verify(graphSearchService, never()).findRelatedEntities(any(), any(), any(), any(), anyInt(), any());

        MLGraphSearchResponse response = captor.getValue();
        assertEquals(1, response.getEntities().size());
        assertEquals(Integer.valueOf(1), response.getTotalCount());
        assertEquals(QUERY, response.getQuery());
    }

    @Test
    public void testDoExecute_TextSearch_ServiceFailure() {
        mockGetContainer(graphEnabledContainer);
        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphEnabledContainer))).thenReturn(true);

        Exception expected = new RuntimeException("embedding failed");
        doAnswer(invocation -> {
            ActionListener<List<GraphEntity>> listener = invocation.getArgument(5);
            listener.onFailure(expected);
            return null;
        }).when(graphSearchService).searchEntitiesByText(eq(QUERY), any(), any(), any(), eq(5), any());

        action.doExecute(task, request(baseInput("text")), actionListener);

        verify(actionListener).onFailure(expected);
    }

    @Test
    public void testDoExecute_TextSearch_DefaultTopKWhenNull() {
        mockGetContainer(graphEnabledContainer);
        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphEnabledContainer))).thenReturn(true);

        MLGraphSearchInput input = MLGraphSearchInput.builder()
            .memoryContainerId(CONTAINER_ID)
            .query(QUERY)
            .searchType("text")
            .topK(null)
            .build();

        doAnswer(invocation -> {
            ActionListener<List<GraphEntity>> listener = invocation.getArgument(5);
            listener.onResponse(List.of());
            return null;
        }).when(graphSearchService).searchEntitiesByText(eq(QUERY), any(), any(), any(), eq(10), any());

        action.doExecute(task, request(input), actionListener);

        verify(graphSearchService, times(1)).searchEntitiesByText(eq(QUERY), any(), any(), any(), eq(10), any());
    }

    @Test
    public void testDoExecute_TraversalSearch_MissingEntityId() {
        mockGetContainer(graphEnabledContainer);
        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphEnabledContainer))).thenReturn(true);

        MLGraphSearchInput input = MLGraphSearchInput.builder()
            .memoryContainerId(CONTAINER_ID)
            .query(QUERY)
            .searchType("traversal")
            .build();

        action.doExecute(task, request(input), actionListener);

        ArgumentCaptor<Exception> captor = ArgumentCaptor.forClass(Exception.class);
        verify(actionListener).onFailure(captor.capture());
        assertTrue(captor.getValue() instanceof OpenSearchStatusException);
        assertEquals(RestStatus.BAD_REQUEST, ((OpenSearchStatusException) captor.getValue()).status());
        assertTrue(captor.getValue().getMessage().contains("Entity ID is required"));
        verify(graphSearchService, never()).findRelatedEntities(any(), any(), any(), any(), anyInt(), any());
    }

    @Test
    public void testDoExecute_TraversalSearch_Success() {
        mockGetContainer(graphEnabledContainer);
        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphEnabledContainer))).thenReturn(true);

        MLGraphSearchInput input = MLGraphSearchInput.builder()
            .memoryContainerId(CONTAINER_ID)
            .query(QUERY)
            .searchType("traversal")
            .entityId("root-entity")
            .maxDepth(3)
            .build();

        GraphEntity entity = GraphEntity.builder().entityId("root-entity").name("Alice").type("person").build();
        GraphRelationship rel = GraphRelationship.builder()
            .relationshipId("r1")
            .sourceEntityId("root-entity")
            .targetEntityId("e2")
            .relationshipType("knows")
            .build();
        GraphTraversalResult traversal = GraphTraversalResult.builder()
            .entities(List.of(entity))
            .relationships(List.of(rel))
            .centerEntityId("root-entity")
            .traversalDepth(3)
            .build();

        doAnswer(invocation -> {
            ActionListener<GraphTraversalResult> listener = invocation.getArgument(5);
            listener.onResponse(traversal);
            return null;
        }).when(graphSearchService).findRelatedEntities(eq("root-entity"), any(), any(), any(), eq(3), any());

        action.doExecute(task, request(input), actionListener);

        ArgumentCaptor<MLGraphSearchResponse> captor = ArgumentCaptor.forClass(MLGraphSearchResponse.class);
        verify(actionListener, times(1)).onResponse(captor.capture());
        MLGraphSearchResponse response = captor.getValue();
        assertEquals(1, response.getEntities().size());
        assertEquals(1, response.getRelationships().size());
        assertEquals(Integer.valueOf(1), response.getTotalCount());
    }

    @Test
    public void testDoExecute_TraversalSearch_DefaultMaxDepthWhenNull() {
        mockGetContainer(graphEnabledContainer);
        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphEnabledContainer))).thenReturn(true);

        MLGraphSearchInput input = MLGraphSearchInput.builder()
            .memoryContainerId(CONTAINER_ID)
            .query(QUERY)
            .searchType("traversal")
            .entityId("root-entity")
            .maxDepth(null)
            .build();

        doAnswer(invocation -> {
            ActionListener<GraphTraversalResult> listener = invocation.getArgument(5);
            listener.onResponse(GraphTraversalResult.builder().entities(List.of()).relationships(List.of()).build());
            return null;
        }).when(graphSearchService).findRelatedEntities(eq("root-entity"), any(), any(), any(), eq(2), any());

        action.doExecute(task, request(input), actionListener);

        verify(graphSearchService, times(1)).findRelatedEntities(eq("root-entity"), any(), any(), any(), eq(2), any());
    }

    @Test
    public void testDoExecute_TraversalSearch_ServiceFailure() {
        mockGetContainer(graphEnabledContainer);
        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphEnabledContainer))).thenReturn(true);

        Exception expected = new RuntimeException("traversal failed");
        doAnswer(invocation -> {
            ActionListener<GraphTraversalResult> listener = invocation.getArgument(5);
            listener.onFailure(expected);
            return null;
        }).when(graphSearchService).findRelatedEntities(eq("root-entity"), any(), any(), any(), anyInt(), any());

        MLGraphSearchInput input = MLGraphSearchInput.builder()
            .memoryContainerId(CONTAINER_ID)
            .query(QUERY)
            .searchType("traversal")
            .entityId("root-entity")
            .build();

        action.doExecute(task, request(input), actionListener);

        verify(actionListener).onFailure(expected);
    }

    @Test
    public void testDoExecute_HybridSearch_Success() {
        mockGetContainer(graphEnabledContainer);
        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphEnabledContainer))).thenReturn(true);

        GraphEntity entity = GraphEntity.builder().entityId("e1").name("Alice").type("person").build();
        GraphSearchResult result = GraphSearchResult.builder()
            .entity(entity)
            .score(0.9f)
            .matchType("hybrid")
            .queryText(QUERY)
            .build();
        doAnswer(invocation -> {
            ActionListener<List<GraphSearchResult>> listener = invocation.getArgument(5);
            listener.onResponse(List.of(result));
            return null;
        }).when(graphSearchService).searchGraphHybrid(eq(QUERY), any(), any(), any(), eq(5), any());

        action.doExecute(task, request(baseInput("hybrid")), actionListener);

        ArgumentCaptor<MLGraphSearchResponse> captor = ArgumentCaptor.forClass(MLGraphSearchResponse.class);
        verify(actionListener, times(1)).onResponse(captor.capture());

        MLGraphSearchResponse response = captor.getValue();
        assertEquals(1, response.getEntities().size());
        assertEquals(1, response.getSearchResults().size());
        assertEquals(Integer.valueOf(1), response.getTotalCount());
    }

    @Test
    public void testDoExecute_UnknownSearchType_FallsBackToHybrid() {
        mockGetContainer(graphEnabledContainer);
        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphEnabledContainer))).thenReturn(true);

        doAnswer(invocation -> {
            ActionListener<List<GraphSearchResult>> listener = invocation.getArgument(5);
            listener.onResponse(List.of());
            return null;
        }).when(graphSearchService).searchGraphHybrid(eq(QUERY), any(), any(), any(), anyInt(), any());

        action.doExecute(task, request(baseInput("bogus-type")), actionListener);

        verify(graphSearchService, times(1)).searchGraphHybrid(eq(QUERY), any(), any(), any(), anyInt(), any());
        verify(graphSearchService, never()).searchEntitiesByText(any(), any(), any(), any(), anyInt(), any());
        verify(graphSearchService, never()).findRelatedEntities(any(), any(), any(), any(), anyInt(), any());
    }

    @Test
    public void testDoExecute_HybridSearch_ServiceFailure() {
        mockGetContainer(graphEnabledContainer);
        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphEnabledContainer))).thenReturn(true);

        Exception expected = new RuntimeException("hybrid failed");
        doAnswer(invocation -> {
            ActionListener<List<GraphSearchResult>> listener = invocation.getArgument(5);
            listener.onFailure(expected);
            return null;
        }).when(graphSearchService).searchGraphHybrid(eq(QUERY), any(), any(), any(), anyInt(), any());

        action.doExecute(task, request(baseInput("hybrid")), actionListener);

        verify(actionListener).onFailure(expected);
    }

    @Test
    public void testDoExecute_NamespaceContainsContainerAndTenant() {
        mockGetContainer(graphEnabledContainer);
        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphEnabledContainer))).thenReturn(true);

        @SuppressWarnings("unchecked")
        ArgumentCaptor<Map<String, String>> namespaceCaptor = ArgumentCaptor.forClass(Map.class);

        doAnswer(invocation -> {
            ActionListener<List<GraphSearchResult>> listener = invocation.getArgument(5);
            listener.onResponse(List.of());
            return null;
        }).when(graphSearchService).searchGraphHybrid(eq(QUERY), any(), namespaceCaptor.capture(), any(), anyInt(), any());

        action.doExecute(task, request(baseInput("hybrid")), actionListener);

        Map<String, String> namespace = namespaceCaptor.getValue();
        assertEquals(CONTAINER_ID, namespace.get("memory_container_id"));
        assertEquals(TENANT_ID, namespace.get("tenant_id"));
        // user is null (no context), so owner_id must be absent
        assertFalse(namespace.containsKey("owner_id"));
    }
}
