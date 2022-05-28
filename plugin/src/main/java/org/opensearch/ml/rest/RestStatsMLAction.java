/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.indices.MLIndicesHandler.ML_MODEL_INDEX;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.client.node.NodeClient;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.action.stats.MLStatsNodeResponse;
import org.opensearch.ml.action.stats.MLStatsNodesAction;
import org.opensearch.ml.action.stats.MLStatsNodesRequest;
import org.opensearch.ml.stats.MLClusterLevelStat;
import org.opensearch.ml.stats.MLNodeLevelStat;
import org.opensearch.ml.stats.MLStatLevel;
import org.opensearch.ml.stats.MLStats;
import org.opensearch.ml.stats.MLStatsInput;
import org.opensearch.ml.utils.IndexUtils;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.BytesRestResponse;
import org.opensearch.rest.RestChannel;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.RestStatus;

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
     * @param mlStats MLStats object
     * @param clusterService cluster service
     * @param indexUtils index util
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
        return ImmutableList
            .of(
                new Route(RestRequest.Method.GET, ML_BASE_URI + "/{nodeId}/stats/"),
                new Route(RestRequest.Method.GET, ML_BASE_URI + "/{nodeId}/stats/{stat}"),
                new Route(RestRequest.Method.GET, ML_BASE_URI + "/stats/"),
                new Route(RestRequest.Method.GET, ML_BASE_URI + "/stats/{stat}")
            );
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        boolean hasContent = request.hasContent();
        MLStatsInput mlStatsInput;
        if (hasContent) {
            XContentParser parser = request.contentParser();
            ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
            mlStatsInput = MLStatsInput.parse(parser);
        } else {
            mlStatsInput = createMlStatsInputFromRequestParams(request);
        }

        String[] nodeIds = mlStatsInput.retrieveStatsOnAllNodes() ? getAllNodes() : mlStatsInput.getNodeIds().toArray(new String[0]);
        MLStatsNodesRequest mlStatsNodesRequest = new MLStatsNodesRequest(nodeIds, mlStatsInput);

        Map<MLClusterLevelStat, Object> clusterStatsMap = new HashMap<>();
        if (mlStatsInput.getTargetStatLevels().contains(MLStatLevel.CLUSTER)) {
            clusterStatsMap.putAll(getClusterStatsMap(mlStatsInput));
        }

        // copy to make a effectively final temp variable finalMlStatsInput
        MLStatsInput finalMlStatsInput = mlStatsInput;
        return channel -> {
            if (finalMlStatsInput.getTargetStatLevels().contains(MLStatLevel.CLUSTER)
                && (finalMlStatsInput.retrieveAllClusterLevelStats()
                    || finalMlStatsInput.getClusterLevelStats().contains(MLClusterLevelStat.ML_MODEL_COUNT))) {
                indexUtils.getNumberOfDocumentsInIndex(ML_MODEL_INDEX, ActionListener.wrap(count -> {
                    clusterStatsMap.put(MLClusterLevelStat.ML_MODEL_COUNT, count);
                    getNodeStats(finalMlStatsInput, clusterStatsMap, client, mlStatsNodesRequest, channel);
                }, e -> {
                    String errorMessage = "Failed to get ML model count";
                    log.error(errorMessage, e);
                    onFailure(channel, RestStatus.INTERNAL_SERVER_ERROR, errorMessage, e);
                }));
            } else {
                getNodeStats(finalMlStatsInput, clusterStatsMap, client, mlStatsNodesRequest, channel);
            }
        };
    }

    MLStatsInput createMlStatsInputFromRequestParams(RestRequest request) {
        MLStatsInput mlStatsInput = new MLStatsInput();
        Optional<String[]> nodeIds = splitCommaSeparatedParam(request, "nodeId");
        if (nodeIds.isPresent()) {
            mlStatsInput.getNodeIds().addAll(Arrays.asList(nodeIds.get()));
        }
        Optional<String[]> stats = splitCommaSeparatedParam(request, "stat");
        if (stats.isPresent()) {
            for (String state : stats.get()) {
                state = state.toUpperCase(Locale.ROOT);
                // only support cluster and node level stats for bwc
                if (state.startsWith("ML_NODE")) {
                    mlStatsInput.getNodeLevelStats().add(MLNodeLevelStat.from(state));
                } else {
                    mlStatsInput.getClusterLevelStats().add(MLClusterLevelStat.from(state));
                }
            }
            if (mlStatsInput.getClusterLevelStats().size() > 0) {
                mlStatsInput.getTargetStatLevels().add(MLStatLevel.CLUSTER);
            }
            if (mlStatsInput.getNodeLevelStats().size() > 0) {
                mlStatsInput.getTargetStatLevels().add(MLStatLevel.NODE);
            }
        } else {
            mlStatsInput.getTargetStatLevels().addAll(EnumSet.allOf(MLStatLevel.class));
        }
        return mlStatsInput;
    }

    void getNodeStats(
        MLStatsInput mlStatsInput,
        Map<MLClusterLevelStat, Object> clusterStatsMap,
        NodeClient client,
        MLStatsNodesRequest mlStatsNodesRequest,
        RestChannel channel
    ) throws IOException {
        XContentBuilder builder = channel.newBuilder();
        if (mlStatsInput.onlyRetrieveClusterLevelStats()) {
            // only return cluster level stats
            builder.startObject();
            if (clusterStatsMap != null && clusterStatsMap.size() > 0) {
                for (Map.Entry<MLClusterLevelStat, Object> entry : clusterStatsMap.entrySet()) {
                    builder.field(entry.getKey().name().toLowerCase(Locale.ROOT), entry.getValue());
                }
            }
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
        } else {
            // retrieve node level stats
            client.execute(MLStatsNodesAction.INSTANCE, mlStatsNodesRequest, ActionListener.wrap(r -> {
                builder.startObject();
                // cluster level stats
                if (clusterStatsMap != null && clusterStatsMap.size() > 0) {
                    for (Map.Entry<MLClusterLevelStat, Object> entry : clusterStatsMap.entrySet()) {
                        builder.field(entry.getKey().name().toLowerCase(Locale.ROOT), entry.getValue());
                    }
                }
                // node level stats: include algorithm and action level stats
                List<MLStatsNodeResponse> nodeStats = r.getNodes().stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
                if (nodeStats != null && nodeStats.size() > 0) {
                    // builder.startObject("nodes");
                    r.toXContent(builder, ToXContent.EMPTY_PARAMS);
                    // builder.endObject();
                }
                builder.endObject();
                channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
            }, e -> {
                String errorMessage = "Failed to get ML node level stats";
                log.error(errorMessage, e);
                onFailure(channel, RestStatus.INTERNAL_SERVER_ERROR, errorMessage, e);
            }));
        }
    }

    private String[] getAllNodes() {
        Iterator<DiscoveryNode> iterator = clusterService.state().nodes().iterator();
        List<String> nodeIds = new ArrayList<>();
        while (iterator.hasNext()) {
            nodeIds.add(iterator.next().getId());
        }
        return nodeIds.toArray(new String[0]);
    }

    private void onFailure(RestChannel channel, RestStatus status, String errorMessage, Exception exception) {
        BytesRestResponse bytesRestResponse;
        try {
            bytesRestResponse = new BytesRestResponse(channel, exception);
        } catch (Exception e) {
            bytesRestResponse = new BytesRestResponse(status, errorMessage);
        }
        channel.sendResponse(bytesRestResponse);
    }

    private Map<MLClusterLevelStat, Object> getClusterStatsMap(MLStatsInput mlStatsInput) {
        Map<MLClusterLevelStat, Object> clusterStats = new HashMap<>();
        mlStats
            .getClusterStats()
            .entrySet()
            .stream()
            .filter(s -> mlStatsInput.retrieveStat(s.getKey()))
            .forEach(s -> clusterStats.put((MLClusterLevelStat) s.getKey(), s.getValue().getValue()));
        return clusterStats;
    }

    @VisibleForTesting
    Optional<String[]> splitCommaSeparatedParam(RestRequest request, String paramName) {
        return Optional.ofNullable(request.param(paramName)).map(s -> s.split(","));
    }
}
