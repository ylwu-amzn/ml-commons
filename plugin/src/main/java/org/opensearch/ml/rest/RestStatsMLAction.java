/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.ml.indices.MLIndicesHandler.ML_MODEL_INDEX;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;
import static org.opensearch.ml.stats.StatNames.ML_MODEL_COUNT;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import lombok.extern.log4j.Log4j;
import lombok.extern.log4j.Log4j2;
import org.opensearch.action.ActionListener;
import org.opensearch.action.admin.indices.stats.IndicesStatsRequest;
import org.opensearch.client.node.NodeClient;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.ml.action.stats.MLStatsNodeResponse;
import org.opensearch.ml.action.stats.MLStatsNodesAction;
import org.opensearch.ml.action.stats.MLStatsNodesRequest;
import org.opensearch.ml.stats.MLStats;
import org.opensearch.ml.stats.StatNames;
import org.opensearch.ml.utils.IndexUtils;
import org.opensearch.plugins.ActionPlugin;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.BytesRestResponse;
import org.opensearch.rest.RestChannel;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.RestStatus;
import org.opensearch.rest.action.RestToXContentListener;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;

//TODO: add IT
@Log4j2
public class RestStatsMLAction extends BaseRestHandler {
    private static final String STATS_ML_ACTION = "stats_ml";
    private MLStats mlStats;
    private ClusterService clusterService;
    private IndexUtils indexUtils;

    /**
     * Constructor
     *
     * @param mlStats MLStats object
     */
    public RestStatsMLAction(MLStats mlStats, ClusterService clusterService, IndexUtils indexUtils) {
        this.mlStats = mlStats;
        this.clusterService = clusterService;
        this.indexUtils = indexUtils;
    }

    @Override
    public String getName() {
        return STATS_ML_ACTION;
    }

    @Override
    public List<Route> routes() {
        /**
         * We are going to have different levels of stats: cluster, node, algorithm, action.
         * User can use "_plugins/_ml/stats" to get all stats on all levels. Or user can get
         * stats on specific level.
         * 1. Cluster level
         *    _plugins/_ml/stats/cluster
         *    or you can specify concrete stats names
         *    _plugins/_ml/stats/cluster/ml_model_count,ml_model_index_status
         * 2. Node level
         *    _plugins/_ml/stats/node
         *    or you can specify concrete stats names
         *    _plugins/_ml/stats/node/ml_total_request_count,ml_total_failure_count
         *    or you can specify concrete stats names on specific node
         *    _plugins/_ml/stats/node/nKWwVPHYSVm5tG1HpfSZMg/ml_total_request_count,ml_total_failure_count
         */
        return ImmutableList
            .of(
                new Route(RestRequest.Method.GET, ML_BASE_URI + "/stats/"),
                new Route(RestRequest.Method.GET, ML_BASE_URI + "/stats/{stat}"),
                new Route(RestRequest.Method.GET, ML_BASE_URI + "/{nodeId}/stats/"),
                new Route(RestRequest.Method.GET, ML_BASE_URI + "/{nodeId}/stats/{stat}"),
//                new Route(RestRequest.Method.GET, ML_BASE_URI + "/algorithms/{algorithm}/stats/"),
//                new Route(RestRequest.Method.GET, ML_BASE_URI + "/algorithms/{algorithm}/stats/{stat}"),
                new Route(RestRequest.Method.GET, ML_BASE_URI + "/{nodeId}/{algorithm}/stats/"),
                new Route(RestRequest.Method.GET, ML_BASE_URI + "/{nodeId}/{algorithm}/stats/{stat}")
            );
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) {
        MLStatsNodesRequest mlStatsNodesRequest = getRequest(request);

        Set<String> statsToBeRetrieved = mlStatsNodesRequest.getStatsToBeRetrieved();
        boolean retrieveAllStats = mlStatsNodesRequest.isRetrieveAllStats();
        Map<String, Object> clusterStatsMap = getClusterStatsMap(statsToBeRetrieved, retrieveAllStats);

        return channel -> {
            if (retrieveAllStats || statsToBeRetrieved.contains(ML_MODEL_COUNT)) {
                indexUtils.getNumberOfDocumentsInIndex(ML_MODEL_INDEX, ActionListener.wrap(count -> {
                    clusterStatsMap.put(ML_MODEL_COUNT, count);
                    getNodeStats(clusterStatsMap, client, mlStatsNodesRequest, channel);
                }, e-> {
                    String errorMessage = "Failed to get ML model count";
                    log.error(errorMessage, e);
                    onFailure(channel, RestStatus.INTERNAL_SERVER_ERROR, errorMessage, e);
                }));
            } else {
                getNodeStats(clusterStatsMap, client, mlStatsNodesRequest, channel);
            }
        };
    }

    private void getNodeStats(Map<String, Object> clusterStatsMap, NodeClient client, MLStatsNodesRequest mlStatsNodesRequest, RestChannel channel) {
        client.execute(MLStatsNodesAction.INSTANCE, mlStatsNodesRequest, ActionListener.wrap(r->{
            XContentBuilder builder = channel.newBuilder();
            builder.startObject();
            // cluster level stats
            if (clusterStatsMap != null && clusterStatsMap.size() > 0) {
                for (Map.Entry<String, Object> entry : clusterStatsMap.entrySet()) {
                    builder.field(entry.getKey(), entry.getValue());
                }
            }
            // node level stats
            List<MLStatsNodeResponse> nodeStats = r.getNodes().stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
            if (nodeStats != null && nodeStats.size() > 0) {
//                builder.startObject("nodes");
                r.toXContent(builder, ToXContent.EMPTY_PARAMS);
//                builder.endObject();
            }
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
        }, e->{
            String errorMessage = "Failed to get ML node level stats";
            log.error(errorMessage, e);
            onFailure(channel, RestStatus.INTERNAL_SERVER_ERROR, errorMessage, e);
        }));
    }

    private void onFailure(RestChannel channel, RestStatus status, String errorMessage, Exception exception) {
        BytesRestResponse bytesRestResponse;
        try{
            bytesRestResponse = new BytesRestResponse(channel, exception);
        } catch (Exception e) {
            bytesRestResponse = new BytesRestResponse(status, errorMessage);
        }
        channel.sendResponse(bytesRestResponse);
    }

    private Map<String, Object> getClusterStatsMap(Set<String> statsToBeRetrieved, boolean retrieveAllStats) {
        Map<String, Object> clusterStats = new HashMap<>();
        mlStats
                .getClusterStats()
                .entrySet()
                .stream()
                .filter(s -> retrieveAllStats || statsToBeRetrieved.contains(s.getKey()))
                .forEach(s -> clusterStats.put(s.getKey(), s.getValue().getValue()));
        return clusterStats;
    }

    /**
     * Creates a MLStatsNodesRequest from a RestRequest
     *
     * @param request RestRequest
     * @return MLStatsNodesRequest
     */
    @VisibleForTesting
    MLStatsNodesRequest getRequest(RestRequest request) {
        // todo: add logic to triage request based on node type(ML node or data node)
        String[] nodeIds = splitCommaSeparatedParam(request, "nodeId").orElse(null);
        String[] algorithms = splitCommaSeparatedParam(request, "algorithm").orElse(null);
        MLStatsNodesRequest mlStatsRequest = new MLStatsNodesRequest(nodeIds, algorithms);
        mlStatsRequest.timeout(request.param("timeout"));

        List<String> requestedStats = splitCommaSeparatedParam(request, "stat").map(Arrays::asList).orElseGet(Collections::emptyList);

        Set<String> validStats = new HashSet<>();
        validStats.addAll(mlStats.getStats().keySet());
        validStats.add("algorithms");
        if (isAllStatsRequested(requestedStats)) {
            mlStatsRequest.setRetrieveAllStats(true);
        } else {
            mlStatsRequest.addAll(getStatsToBeRetrieved(request, validStats, requestedStats));
        }

        return mlStatsRequest;
    }

    @VisibleForTesting
    Set<String> getStatsToBeRetrieved(RestRequest request, Set<String> validStats, List<String> requestedStats) {
        if (requestedStats.contains(MLStatsNodesRequest.ALL_STATS_KEY)) {
            throw new IllegalArgumentException(
                String
                    .format(
                        Locale.ROOT,
                        "Request %s contains both %s and individual stats",
                        request.path(),
                        MLStatsNodesRequest.ALL_STATS_KEY
                    )
            );
        }

        Set<String> invalidStats = requestedStats.stream().filter(s -> !validStats.contains(s)).collect(Collectors.toSet());

        if (!invalidStats.isEmpty()) {
            throw new IllegalArgumentException(unrecognized(request, invalidStats, new HashSet<>(requestedStats), "stat"));
        }
        return new HashSet<>(requestedStats);
    }

    @VisibleForTesting
    boolean isAllStatsRequested(List<String> requestedStats) {
        return requestedStats.isEmpty() || (requestedStats.size() == 1 && requestedStats.contains(MLStatsNodesRequest.ALL_STATS_KEY));
    }

    @VisibleForTesting
    Optional<String[]> splitCommaSeparatedParam(RestRequest request, String paramName) {
        String nodeId = request.param(paramName);
        if ("_all".equals(nodeId)) {
            Iterator<DiscoveryNode> iterator = clusterService.state().getNodes().iterator();
            List<String> allNodes = new ArrayList<>();
            while (iterator.hasNext()) {
                allNodes.add(iterator.next().getId());
            }
            return Optional.ofNullable(allNodes.toArray(new String[0]));
        }
        return Optional.ofNullable(nodeId).map(s -> s.split(","));
    }
}
