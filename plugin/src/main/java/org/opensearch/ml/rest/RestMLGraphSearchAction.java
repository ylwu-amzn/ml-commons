/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.PARAMETER_MEMORY_CONTAINER_ID;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.core.rest.RestStatus;
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
            throw new OpenSearchStatusException("Agentic memory feature is not enabled", RestStatus.FORBIDDEN);
        }

        String memoryContainerId = request.param(PARAMETER_MEMORY_CONTAINER_ID);
        String tenantId = TenantAwareHelper.getTenantID(mlFeatureEnabledSetting.isMultiTenancyEnabled(), request);

        if (!request.hasContent()) {
            throw new IllegalArgumentException("Request body is required for graph search");
        }
        XContentParser parser = request.contentParser();
        MLGraphSearchRequest parsedRequest = MLGraphSearchRequest.parse(parser, tenantId);

        // Override memory container ID from URL path if not provided in body
        final MLGraphSearchRequest mlGraphSearchRequest;
        if (parsedRequest.getInput().getMemoryContainerId() == null) {
            MLGraphSearchInput updatedInput = MLGraphSearchInput.builder()
                .memoryContainerId(memoryContainerId)
                .query(parsedRequest.getInput().getQuery())
                .topK(parsedRequest.getInput().getTopK())
                .entityId(parsedRequest.getInput().getEntityId())
                .maxDepth(parsedRequest.getInput().getMaxDepth())
                .searchType(parsedRequest.getInput().getSearchType())
                .entityType(parsedRequest.getInput().getEntityType())
                .build();

            mlGraphSearchRequest = MLGraphSearchRequest.builder()
                .input(updatedInput)
                .tenantId(tenantId)
                .build();
        } else {
            mlGraphSearchRequest = parsedRequest;
        }

        return channel -> client.execute(
            MLGraphSearchAction.INSTANCE,
            mlGraphSearchRequest,
            new RestToXContentListener<>(channel)
        );
    }
}