/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.core.action.ActionListener;
import org.opensearch.ml.common.transport.memory.MLGraphSearchInput;
import org.opensearch.ml.common.transport.memory.MLGraphSearchRequest;
import org.opensearch.ml.common.transport.memory.MLGraphSearchResponse;
import org.opensearch.tasks.Task;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.transport.TransportService;

public class TransportListGraphEntitiesActionTests extends OpenSearchTestCase {

    @Mock
    private TransportService transportService;

    @Mock
    private ActionFilters actionFilters;

    @Mock
    private TransportGraphSearchAction graphSearchAction;

    @Mock
    private Task task;

    @Mock
    private ActionListener<MLGraphSearchResponse> actionListener;

    private TransportListGraphEntitiesAction transportListGraphEntitiesAction;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        MockitoAnnotations.openMocks(this);

        transportListGraphEntitiesAction = new TransportListGraphEntitiesAction(
            transportService,
            actionFilters,
            graphSearchAction
        );
    }

    @Test
    public void testDoExecute_DelegatesToGraphSearchAction() {
        MLGraphSearchInput input = MLGraphSearchInput.builder()
            .memoryContainerId("container-123")
            .query("*")
            .searchType("text")
            .build();
        MLGraphSearchRequest request = MLGraphSearchRequest.builder()
            .input(input)
            .tenantId("tenant-1")
            .build();

        transportListGraphEntitiesAction.doExecute(task, request, actionListener);

        verify(graphSearchAction, times(1)).doExecute(task, request, actionListener);
    }
}
