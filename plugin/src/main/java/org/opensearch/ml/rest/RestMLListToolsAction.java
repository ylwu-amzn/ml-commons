package org.opensearch.ml.rest;

import com.google.common.collect.ImmutableList;
import org.opensearch.client.node.NodeClient;
import org.opensearch.ml.common.ToolMetadata;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.transport.tools.MLListToolsAction;
import org.opensearch.ml.common.transport.tools.MLToolsListRequest;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;

public class RestMLListToolsAction extends BaseRestHandler {
    private static final String ML_GET_MODEL_ACTION = "ml_get_tools_action";

    private Map<String, Tool> externalTools;

    public RestMLListToolsAction(Map<String, Tool> externalTools) {
        this.externalTools = externalTools;
    }

    @Override
    public String getName() {
        return ML_GET_MODEL_ACTION;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList
                .of(new Route(RestRequest.Method.GET, String.format(Locale.ROOT, "%s/tools", ML_BASE_URI)));
    }

    /**
     * Prepare the request for execution. Implementations should consume all request params before
     * returning the runnable for actual execution. Unconsumed params will immediately terminate
     * execution of the request. However, some params are only used in processing the response;
     * implementations can override {@link BaseRestHandler#responseParams()} to indicate such
     * params.
     *
     * @param request the request to execute
     * @param client  client for executing actions on the local node
     * @return the action to execute
     * @throws IOException if an I/O exception occurred parsing the request and preparing for
     *                     execution
     */
    @Override
    protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        List<ToolMetadata> toolList = new ArrayList<>();
        externalTools.forEach((key, value) -> toolList.add(
                ToolMetadata.builder()
                        .name(value.getName())
                        .description(value.getDescription())
                        .build()));
        MLToolsListRequest mlToolsListRequest = MLToolsListRequest.builder().externalTools(toolList).build();
        return channel -> client.execute(MLListToolsAction.INSTANCE, mlToolsListRequest, new RestToXContentListener<>(channel));
    }
}
