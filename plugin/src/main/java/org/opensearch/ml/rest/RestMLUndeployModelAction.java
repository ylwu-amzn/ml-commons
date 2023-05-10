/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;
import static org.opensearch.ml.utils.RestActionUtils.PARAMETER_MODEL_ID;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;

import org.apache.commons.lang3.ArrayUtils;
import org.opensearch.client.node.NodeClient;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.transport.undeploy.MLUndeployModelsAction;
import org.opensearch.ml.common.transport.undeploy.MLUndeployModelsRequest;
import org.opensearch.ml.common.transport.undeploy.MLUndeployModelInput;
import org.opensearch.ml.common.transport.undeploy.MLUndeployModelNodesRequest;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;

public class RestMLUndeployModelAction extends BaseRestHandler {
    private static final String ML_UNDEPLOY_MODEL_ACTION = "ml_undeploy_model_action";
    private ClusterService clusterService;

    /**
     * Constructor
     */
    public RestMLUndeployModelAction(ClusterService clusterService) {
        this.clusterService = clusterService;
    }

    @Override
    public String getName() {
        return ML_UNDEPLOY_MODEL_ACTION;
    }

    @Override
    public List<ReplacedRoute> replacedRoutes() {
        return ImmutableList
            .of(
                new ReplacedRoute(
                    RestRequest.Method.POST,
                    String.format(Locale.ROOT, "%s/models/{%s}/_undeploy", ML_BASE_URI, PARAMETER_MODEL_ID),// new url
                    RestRequest.Method.POST,
                    String.format(Locale.ROOT, "%s/models/{%s}/_unload", ML_BASE_URI, PARAMETER_MODEL_ID)// old url
                ),
                new ReplacedRoute(
                    RestRequest.Method.POST,
                    String.format(Locale.ROOT, "%s/models/_undeploy", ML_BASE_URI),// new url
                    RestRequest.Method.POST,
                    String.format(Locale.ROOT, "%s/models/_unload", ML_BASE_URI)// old url
                )
            );
    }

    @Override
    public RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        //MLUndeployModelNodesRequest MLUndeployModelNodesRequest = getRequest(request);
        MLUndeployModelsRequest mlUndeployModelsRequest = getUndeployRequest(request);
        return channel -> client
//            .execute(MLUndeployModelAction.INSTANCE, MLUndeployModelNodesRequest, new RestToXContentListener<>(channel));
            .execute(MLUndeployModelsAction.INSTANCE, mlUndeployModelsRequest, new RestToXContentListener<>(channel));
    }

    MLUndeployModelsRequest getUndeployRequest(RestRequest request) throws IOException {
        String modelId = request.param(PARAMETER_MODEL_ID);
        String[] targetModelIds = null;
        if (modelId != null) {
            targetModelIds = new String[] { modelId };
        }
        String[] targetNodeIds = getAllNodes();
        if (request.hasContent()) {
            XContentParser parser = request.contentParser();
            ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
            MLUndeployModelInput mlInput = MLUndeployModelInput.parse(parser);
            String[] nodeIds = mlInput.getNodeIds();
            String[] modelIds = mlInput.getModelIds();

            if (ArrayUtils.isNotEmpty(nodeIds)) {
                targetNodeIds = nodeIds;
            }
            if (ArrayUtils.isNotEmpty(modelIds)) {
                targetModelIds = modelIds;
            }
        }

        return new MLUndeployModelsRequest(targetNodeIds, targetModelIds);
    }

    /**
     * Creates a MLTrainingTaskRequest from a RestRequest
     *
     * @param request RestRequest
     * @return MLTrainingTaskRequest
     */
    @VisibleForTesting
    MLUndeployModelNodesRequest getRequest(RestRequest request) throws IOException {
        String modelId = request.param(PARAMETER_MODEL_ID);
        String[] targetModelIds = null;
        if (modelId != null) {
            targetModelIds = new String[] { modelId };
        }
        String[] targetNodeIds = getAllNodes();
        if (request.hasContent()) {
            XContentParser parser = request.contentParser();
            ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
            MLUndeployModelInput mlInput = MLUndeployModelInput.parse(parser);
            String[] nodeIds = mlInput.getNodeIds();
            String[] modelIds = mlInput.getModelIds();

            if (ArrayUtils.isNotEmpty(nodeIds)) {
                targetNodeIds = nodeIds;
            }
            if (ArrayUtils.isNotEmpty(modelIds)) {
                targetModelIds = modelIds;
            }
        }

        return new MLUndeployModelNodesRequest(targetNodeIds, targetModelIds);
    }

    private String[] getAllNodes() {
        Iterator<DiscoveryNode> iterator = clusterService.state().nodes().iterator();
        List<String> nodeIds = new ArrayList<>();
        while (iterator.hasNext()) {
            nodeIds.add(iterator.next().getId());
        }
        return nodeIds.toArray(new String[0]);
    }
}
