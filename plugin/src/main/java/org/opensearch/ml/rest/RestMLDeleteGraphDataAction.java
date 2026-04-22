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
import org.opensearch.ml.common.transport.memory.MLDeleteGraphDataAction;
import org.opensearch.ml.common.transport.memory.MLDeleteGraphDataRequest;
import org.opensearch.ml.utils.TenantAwareHelper;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;
import org.opensearch.transport.client.node.NodeClient;

import com.google.common.collect.ImmutableList;

/**
 * REST handler for deleting graph data
 */
public class RestMLDeleteGraphDataAction extends BaseRestHandler {

    private static final String ML_DELETE_GRAPH_DATA_ACTION = "ml_delete_graph_data_action";
    private final MLFeatureEnabledSetting mlFeatureEnabledSetting;

    public RestMLDeleteGraphDataAction(MLFeatureEnabledSetting mlFeatureEnabledSetting) {
        this.mlFeatureEnabledSetting = mlFeatureEnabledSetting;
    }

    @Override
    public String getName() {
        return ML_DELETE_GRAPH_DATA_ACTION;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList.of(
            new Route(
                RestRequest.Method.DELETE,
                String.format(
                    Locale.ROOT,
                    "%s/memory_containers/{%s}/memories/graph",
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

        MLDeleteGraphDataRequest deleteRequest = MLDeleteGraphDataRequest.builder()
            .memoryContainerId(memoryContainerId)
            .tenantId(tenantId)
            .build();

        return channel -> client.execute(
            MLDeleteGraphDataAction.INSTANCE,
            deleteRequest,
            new RestToXContentListener<>(channel)
        );
    }
}