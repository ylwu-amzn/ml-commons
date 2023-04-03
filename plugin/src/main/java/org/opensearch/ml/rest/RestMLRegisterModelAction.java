/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;
import static org.opensearch.ml.utils.RestActionUtils.PARAMETER_DEPLOY_MODEL;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import org.opensearch.client.node.NodeClient;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.transport.register.MLRegisterModelAction;
import org.opensearch.ml.common.transport.register.MLRegisterModelInput;
import org.opensearch.ml.common.transport.register.MLRegisterModelRequest;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;

public class RestMLRegisterModelAction extends BaseRestHandler {
    private static final String ML_REGISTER_MODEL_ACTION = "ml_register_model_action";

    /**
     * Constructor
     */
    public RestMLRegisterModelAction() {}

    @Override
    public String getName() {
        return ML_REGISTER_MODEL_ACTION;
    }

    @Override
    public List<ReplacedRoute> replacedRoutes() {
        return ImmutableList
            .of(
                new ReplacedRoute(
                    RestRequest.Method.POST,
                    String.format(Locale.ROOT, "%s/models/_register", ML_BASE_URI),// new url
                    RestRequest.Method.POST,
                    String.format(Locale.ROOT, "%s/models/_upload", ML_BASE_URI)// old url
                )
            );
    }

    @Override
    public RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        MLRegisterModelRequest mlRegisterModelRequest = getRequest(request);
        return channel -> client.execute(MLRegisterModelAction.INSTANCE, mlRegisterModelRequest, new RestToXContentListener<>(channel));
    }

    /**
     * Creates a MLTrainingTaskRequest from a RestRequest
     *
     * @param request RestRequest
     * @return MLTrainingTaskRequest
     */
    @VisibleForTesting
    MLRegisterModelRequest getRequest(RestRequest request) throws IOException {
        boolean deployModel = request.paramAsBoolean(PARAMETER_DEPLOY_MODEL, false);

        XContentParser parser = request.contentParser();
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
        MLRegisterModelInput mlInput = MLRegisterModelInput.parse(parser, deployModel);
        return new MLRegisterModelRequest(mlInput);
    }
}
