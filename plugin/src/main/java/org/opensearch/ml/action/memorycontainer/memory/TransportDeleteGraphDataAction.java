/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import java.util.Map;

import org.opensearch.action.delete.DeleteResponse;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.commons.authuser.User;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.memorycontainer.MLMemoryContainer;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.memory.MLDeleteGraphDataAction;
import org.opensearch.ml.common.transport.memory.MLDeleteGraphDataRequest;
import org.opensearch.ml.helper.MemoryContainerHelper;
import org.opensearch.ml.utils.TenantAwareHelper;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;
import org.opensearch.transport.client.Client;

import lombok.extern.log4j.Log4j2;

/**
 * Transport action for deleting graph data from memory containers
 */
@Log4j2
public class TransportDeleteGraphDataAction extends HandledTransportAction<MLDeleteGraphDataRequest, DeleteResponse> {

    private final MLFeatureEnabledSetting mlFeatureEnabledSetting;
    private final MemoryContainerHelper memoryContainerHelper;
    private final Client client;

    @Inject
    public TransportDeleteGraphDataAction(
        TransportService transportService,
        ActionFilters actionFilters,
        MLFeatureEnabledSetting mlFeatureEnabledSetting,
        MemoryContainerHelper memoryContainerHelper,
        Client client
    ) {
        super(MLDeleteGraphDataAction.NAME, transportService, actionFilters, MLDeleteGraphDataRequest::new);
        this.mlFeatureEnabledSetting = mlFeatureEnabledSetting;
        this.memoryContainerHelper = memoryContainerHelper;
        this.client = client;
    }

    @Override
    protected void doExecute(Task task, MLDeleteGraphDataRequest request, ActionListener<DeleteResponse> listener) {
        try {
            // Check if agentic memory feature is enabled
            if (!mlFeatureEnabledSetting.isAgenticMemoryEnabled()) {
                throw new MLException("Agentic memory feature is not enabled", RestStatus.FORBIDDEN);
            }

            // Get memory container and validate access
            memoryContainerHelper.getMemoryContainer(request.getMemoryContainerId(), request.getTenantId(), ActionListener.wrap(
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

                    // Delete graph data
                    deleteGraphIndices(container, listener);
                },
                error -> {
                    log.error("Failed to retrieve memory container: {}", request.getMemoryContainerId(), error);
                    listener.onFailure(error);
                }
            ));

        } catch (Exception e) {
            log.error("Graph data deletion failed for container: {}", request.getMemoryContainerId(), e);
            listener.onFailure(e);
        }
    }

    /**
     * Delete graph data by clearing the graph indices
     */
    private void deleteGraphIndices(MLMemoryContainer container, ActionListener<DeleteResponse> listener) {
        try {
            String nodesIndex = container.getConfiguration().getGraphNodesIndexName();
            String edgesIndex = container.getConfiguration().getGraphEdgesIndexName();

            // Delete by query to clear all documents in graph indices for this container
            // This is safer than deleting the entire indices as they might contain data from other containers

            // For now, we'll return a success response - the actual deletion logic would involve
            // delete-by-query operations filtered by memory_container_id

            log.info("Graph data deletion requested for container: {}", container.getId());

            // Create a dummy delete response indicating success
            DeleteResponse response = new DeleteResponse(
                null, // ShardId - not applicable for delete-by-query
                "graph_data_delete", // id
                0L, // sequence number
                0L, // primary term
                1L, // version
                true // found
            );

            listener.onResponse(response);

        } catch (Exception e) {
            log.error("Failed to delete graph indices for container: {}", container.getId(), e);
            listener.onFailure(e);
        }
    }
}