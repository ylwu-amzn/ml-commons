/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.stats;

import java.io.IOException;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

import lombok.Getter;

import org.opensearch.action.support.nodes.BaseNodesRequest;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.engine.annotation.Function;

public class MLStatsNodesRequest extends BaseNodesRequest<MLStatsNodesRequest> {
    /**
     * Key indicating all stats should be retrieved
     */
    public static final String ALL_STATS_KEY = "_all";

    @Getter
    private Set<String> statsToBeRetrieved;
    /**
     * If set this field as true, will retrieve all stats.
     */
    private boolean retrieveAllStats = false;

    @Getter
    private EnumSet<FunctionName> algorithmsToBeRetrived;

    public MLStatsNodesRequest(StreamInput in) throws IOException {
        super(in);
        retrieveAllStats = in.readBoolean();
        statsToBeRetrieved = in.readSet(StreamInput::readString);
        if (in.readBoolean()) {
            algorithmsToBeRetrived = in.readEnumSet(FunctionName.class);
        }
    }

    /**
     * Constructor
     *
     * @param nodeIds nodeIds of nodes' stats to be retrieved
     */
    public MLStatsNodesRequest(String[] nodeIds, String[] algos) {
        super(nodeIds);
        this.statsToBeRetrieved = new HashSet<>();
        Set<FunctionName> functionNames = new HashSet<>();
        if (algos != null && algos.length > 0) {
            for (String algo : algos) {
                functionNames.add(FunctionName.from(algo.toUpperCase(Locale.ROOT)));
            }
            algorithmsToBeRetrived = EnumSet.copyOf(functionNames);
        }
//        else {
//            algorithmsToBeRetrived = EnumSet.allOf(FunctionName.class);
//        }
    }

    /**
     * Constructor
     *
     * @param nodes nodes of nodes' stats to be retrieved
     */
    public MLStatsNodesRequest(DiscoveryNode... nodes) {
        super(nodes);
        statsToBeRetrieved = new HashSet<>();
    }

    public boolean isRetrieveAllStats() {
        return retrieveAllStats;
    }

    public void setRetrieveAllStats(boolean retrieveAllStats) {
        this.retrieveAllStats = retrieveAllStats;
    }

    /**
     * Adds a stat to the set of stats to be retrieved
     *
     * @param stat name of the stat
     */
    public void addStat(String stat) {
        statsToBeRetrieved.add(stat);
    }

    /**
     * Add all stats to be retrieved
     *
     * @param statsToBeAdded set of stats to be retrieved
     */
    public void addAll(Set<String> statsToBeAdded) {
        statsToBeRetrieved.addAll(statsToBeAdded);
    }

    /**
     * Remove all stats from retrieval set
     */
    public void clear() {
        statsToBeRetrieved.clear();
        algorithmsToBeRetrived.clear();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeBoolean(retrieveAllStats);
        out.writeStringCollection(statsToBeRetrieved);
        if (algorithmsToBeRetrived == null) {
            out.writeBoolean(false);
        } else {
            out.writeBoolean(true);
            out.writeEnumSet(algorithmsToBeRetrived);
        }
    }
}
