/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.PARAMETER_MEMORY_CONTAINER_ID;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;
import static org.opensearch.ml.utils.RestActionUtils.getAllNodes;
import static org.opensearch.ml.utils.RestActionUtils.returnContent;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.memory.MLGraphSearchAction;
import org.opensearch.ml.common.transport.memory.MLGraphSearchInput;
import org.opensearch.ml.common.transport.memory.MLGraphSearchRequest;
import org.opensearch.ml.utils.TenantAwareHelper;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;
import org.opensearch.transport.client.node.NodeClient;

import com.google.common.collect.ImmutableList;

/**
 * REST handler for graph search operations
 */
public class RestMLGraphSearchAction extends BaseRestHandler {

    private static final String ML_GRAPH_SEARCH_ACTION = "ml_graph_search_action";
    private final MLFeatureEnabledSetting mlFeatureEnabledSetting;

    public RestMLGraphSearchAction(MLFeatureEnabledSetting mlFeatureEnabledSetting) {
        this.mlFeatureEnabledSetting = mlFeatureEnabledSetting;
    }

    @Override
    public String getName() {
        return ML_GRAPH_SEARCH_ACTION;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList.of(
            new Route(
                RestRequest.Method.POST,
                String.format(
                    Locale.ROOT,
                    "%s/memory_containers/{%s}/memories/graph/_search",
                    ML_BASE_URI,
                    PARAMETER_MEMORY_CONTAINER_ID
                )
            )
        );
    }

    @Override
    public RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        if (!mlFeatureEnabledSetting.isAgenticMemoryEnabled()) {
            throw new IllegalStateException("Agentic memory feature is not enabled");
        }

        String memoryContainerId = request.param(PARAMETER_MEMORY_CONTAINER_ID);
        String tenantId = TenantAwareHelper.getTenantID(request);

        MLGraphSearchRequest mlGraphSearchRequest;
        if (request.hasContent()) {
            XContentParser parser = request.contentParser();
            mlGraphSearchRequest = MLGraphSearchRequest.parse(parser, tenantId);

            // Override memory container ID from URL path if not provided in body
            if (mlGraphSearchRequest.getInput().getMemoryContainerId() == null) {
                MLGraphSearchInput updatedInput = MLGraphSearchInput.builder()
                    .memoryContainerId(memoryContainerId)
                    .query(mlGraphSearchRequest.getInput().getQuery())
                    .topK(mlGraphSearchRequest.getInput().getTopK())
                    .entityId(mlGraphSearchRequest.getInput().getEntityId())
                    .maxDepth(mlGraphSearchRequest.getInput().getMaxDepth())
                    .searchType(mlGraphSearchRequest.getInput().getSearchType())
                    .entityType(mlGraphSearchRequest.getInput().getEntityType())
                    .build();

                mlGraphSearchRequest = MLGraphSearchRequest.builder()
                    .input(updatedInput)
                    .tenantId(tenantId)
                    .build();
            }
        } else {
            throw new IllegalArgumentException("Request body is required for graph search");
        }

        return channel -> client.execute(
            MLGraphSearchAction.INSTANCE,
            mlGraphSearchRequest,
            new RestToXContentListener<>(channel)
        );
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client, String[] targetNodes) throws IOException {
        // Graph search should run on all nodes for distributed search capability
        return prepareRequest(request, client);
    }
}