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

import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.memory.MLGraphSearchInput;
import org.opensearch.ml.common.transport.memory.MLGraphSearchRequest;
import org.opensearch.ml.common.transport.memory.MLListGraphEntitiesAction;
import org.opensearch.ml.utils.TenantAwareHelper;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;
import org.opensearch.transport.client.node.NodeClient;

import com.google.common.collect.ImmutableList;

/**
 * REST handler for listing graph entities
 */
public class RestMLListGraphEntitiesAction extends BaseRestHandler {

    private static final String ML_LIST_GRAPH_ENTITIES_ACTION = "ml_list_graph_entities_action";
    private final MLFeatureEnabledSetting mlFeatureEnabledSetting;

    public RestMLListGraphEntitiesAction(MLFeatureEnabledSetting mlFeatureEnabledSetting) {
        this.mlFeatureEnabledSetting = mlFeatureEnabledSetting;
    }

    @Override
    public String getName() {
        return ML_LIST_GRAPH_ENTITIES_ACTION;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList.of(
            new Route(
                RestRequest.Method.GET,
                String.format(
                    Locale.ROOT,
                    "%s/memory_containers/{%s}/memories/graph/entities",
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
        String tenantId = TenantAwareHelper.getTenantID(mlFeatureEnabledSetting.isMultiTenancyEnabled(), request);

        // Parse query parameters
        String entityType = request.param("entity_type");
        int topK = request.paramAsInt("top_k", 50);
        String query = request.param("query", "*"); // Default to match all

        MLGraphSearchInput input = MLGraphSearchInput.builder()
            .memoryContainerId(memoryContainerId)
            .query(query)
            .topK(topK)
            .searchType("text")
            .entityType(entityType)
            .build();

        MLGraphSearchRequest mlGraphSearchRequest = MLGraphSearchRequest.builder()
            .input(input)
            .tenantId(tenantId)
            .build();

        return channel -> client.execute(
            MLListGraphEntitiesAction.INSTANCE,
            mlGraphSearchRequest,
            new RestToXContentListener<>(channel)
        );
    }
}