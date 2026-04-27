/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import org.junit.Before;
import org.junit.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.delete.DeleteResponse;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.ml.common.memorycontainer.MLMemoryContainer;
import org.opensearch.ml.common.memorycontainer.MemoryConfiguration;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.memory.MLDeleteGraphDataRequest;
import org.opensearch.ml.helper.MemoryContainerHelper;
import org.opensearch.tasks.Task;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;
import org.opensearch.transport.client.Client;

public class TransportDeleteGraphDataActionTests extends OpenSearchTestCase {

    @Mock
    private TransportService transportService;

    @Mock
    private ActionFilters actionFilters;

    @Mock
    private MLFeatureEnabledSetting mlFeatureEnabledSetting;

    @Mock
    private MemoryContainerHelper memoryContainerHelper;

    @Mock
    private Client client;

    @Mock
    private ThreadPool threadPool;

    @Mock
    private Task task;

    @Mock
    private ActionListener<DeleteResponse> actionListener;

    private ThreadContext threadContext;
    private TransportDeleteGraphDataAction action;
    private MLMemoryContainer graphEnabledContainer;
    private MLMemoryContainer graphDisabledContainer;

    private static final String CONTAINER_ID = "container-123";
    private static final String TENANT_ID = "tenant-1";

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

        action = new TransportDeleteGraphDataAction(
            transportService,
            actionFilters,
            mlFeatureEnabledSetting,
            memoryContainerHelper,
            client
        );
    }

    private MLDeleteGraphDataRequest newRequest() {
        return MLDeleteGraphDataRequest.builder()
            .memoryContainerId(CONTAINER_ID)
            .tenantId(TENANT_ID)
            .build();
    }

    @Test
    public void testDoExecute_FeatureDisabled() {
        when(mlFeatureEnabledSetting.isAgenticMemoryEnabled()).thenReturn(false);

        action.doExecute(task, newRequest(), actionListener);

        ArgumentCaptor<Exception> captor = ArgumentCaptor.forClass(Exception.class);
        verify(actionListener, times(1)).onFailure(captor.capture());
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

        action.doExecute(task, newRequest(), actionListener);

        verify(actionListener, times(1)).onFailure(expected);
        verify(memoryContainerHelper, never()).checkMemoryContainerAccess(any(), any());
    }

    @Test
    public void testDoExecute_AccessDenied() {
        doAnswer(invocation -> {
            ActionListener<MLMemoryContainer> listener = invocation.getArgument(2);
            listener.onResponse(graphEnabledContainer);
            return null;
        }).when(memoryContainerHelper).getMemoryContainer(eq(CONTAINER_ID), eq(TENANT_ID), any());

        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphEnabledContainer))).thenReturn(false);

        action.doExecute(task, newRequest(), actionListener);

        ArgumentCaptor<Exception> captor = ArgumentCaptor.forClass(Exception.class);
        verify(actionListener, times(1)).onFailure(captor.capture());
        assertTrue(captor.getValue() instanceof OpenSearchStatusException);
        assertEquals(RestStatus.FORBIDDEN, ((OpenSearchStatusException) captor.getValue()).status());
        assertTrue(captor.getValue().getMessage().contains(CONTAINER_ID));
    }

    @Test
    public void testDoExecute_GraphDisabled() {
        doAnswer(invocation -> {
            ActionListener<MLMemoryContainer> listener = invocation.getArgument(2);
            listener.onResponse(graphDisabledContainer);
            return null;
        }).when(memoryContainerHelper).getMemoryContainer(eq(CONTAINER_ID), eq(TENANT_ID), any());

        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphDisabledContainer))).thenReturn(true);

        action.doExecute(task, newRequest(), actionListener);

        ArgumentCaptor<Exception> captor = ArgumentCaptor.forClass(Exception.class);
        verify(actionListener, times(1)).onFailure(captor.capture());
        assertTrue(captor.getValue() instanceof OpenSearchStatusException);
        assertEquals(RestStatus.BAD_REQUEST, ((OpenSearchStatusException) captor.getValue()).status());
        assertTrue(captor.getValue().getMessage().contains("Graph functionality is not enabled"));
    }

    @Test
    public void testDoExecute_Success() {
        doAnswer(invocation -> {
            ActionListener<MLMemoryContainer> listener = invocation.getArgument(2);
            listener.onResponse(graphEnabledContainer);
            return null;
        }).when(memoryContainerHelper).getMemoryContainer(eq(CONTAINER_ID), eq(TENANT_ID), any());

        when(memoryContainerHelper.checkMemoryContainerAccess(isNull(), eq(graphEnabledContainer))).thenReturn(true);

        action.doExecute(task, newRequest(), actionListener);

        ArgumentCaptor<DeleteResponse> captor = ArgumentCaptor.forClass(DeleteResponse.class);
        verify(actionListener, times(1)).onResponse(captor.capture());
        verify(actionListener, never()).onFailure(any());

        DeleteResponse response = captor.getValue();
        assertEquals("graph_data_delete", response.getId());
    }

    @Test
    public void testDoExecute_SynchronousExceptionPropagates() {
        when(mlFeatureEnabledSetting.isAgenticMemoryEnabled()).thenThrow(new RuntimeException("boom"));

        action.doExecute(task, newRequest(), actionListener);

        ArgumentCaptor<Exception> captor = ArgumentCaptor.forClass(Exception.class);
        verify(actionListener, times(1)).onFailure(captor.capture());
        assertEquals("boom", captor.getValue().getMessage());
    }
}
