/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import org.opensearch.client.node.NodeClient;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.transport.model.MLModelUpdateAction;
import org.opensearch.ml.common.transport.model.MLModelUpdateRequest;
import org.opensearch.ml.common.transport.register.MLUpdateModelInput;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;
import static org.opensearch.ml.utils.RestActionUtils.PARAMETER_MODEL_ID;
import static org.opensearch.ml.utils.RestActionUtils.getParameterId;

public class RestMLUpdateModelAction extends BaseRestHandler {
    private static final String ML_UPDATE_MODEL_ACTION = "ml_update_model_action";

    /**
     * Constructor
     */
    public RestMLUpdateModelAction() {}

    @Override
    public String getName() {
        return ML_UPDATE_MODEL_ACTION;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList
            .of(new Route(RestRequest.Method.PUT, String.format(Locale.ROOT, "%s/models/{%s}", ML_BASE_URI, PARAMETER_MODEL_ID)));
    }

    @Override
    public RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        MLModelUpdateRequest mlModelUpdateRequest = getRequest(request);
        return channel -> client.execute(MLModelUpdateAction.INSTANCE, mlModelUpdateRequest, new RestToXContentListener<>(channel));
    }

    /**
     * Creates a MLModelGetRequest from a RestRequest
     *
     * @param request RestRequest
     * @return MLModelGetRequest
     */
    @VisibleForTesting
    MLModelUpdateRequest getRequest(RestRequest request) throws IOException {
        String modelId = getParameterId(request, PARAMETER_MODEL_ID);
        XContentParser parser = request.contentParser();
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
        MLUpdateModelInput updateModelInput = MLUpdateModelInput.parse(parser, false);

        return new MLModelUpdateRequest(modelId, updateModelInput);
    }
}
