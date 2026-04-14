/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import org.opensearch.action.ActionListener;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.ml.common.transport.memory.MLGraphSearchRequest;
import org.opensearch.ml.common.transport.memory.MLGraphSearchResponse;
import org.opensearch.ml.common.transport.memory.MLListGraphEntitiesAction;
import org.opensearch.ml.settings.MLFeatureEnabledSetting;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

/**
 * Transport action for listing graph entities - delegates to graph search
 */
public class TransportListGraphEntitiesAction extends HandledTransportAction<MLGraphSearchRequest, MLGraphSearchResponse> {

    private final TransportGraphSearchAction graphSearchAction;

    @Inject
    public TransportListGraphEntitiesAction(
        TransportService transportService,
        ActionFilters actionFilters,
        TransportGraphSearchAction graphSearchAction
    ) {
        super(MLListGraphEntitiesAction.NAME, transportService, actionFilters, MLGraphSearchRequest::new);
        this.graphSearchAction = graphSearchAction;
    }

    @Override
    protected void doExecute(Task task, MLGraphSearchRequest request, ActionListener<MLGraphSearchResponse> listener) {
        // Delegate to graph search action - list entities is just a specialized search
        graphSearchAction.doExecute(task, request, listener);
    }
}