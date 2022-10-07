/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.cluster;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.client.Client;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpInput;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpNodeResponse;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpNodesRequest;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpOnNodeAction;

@Log4j2
public class SyncUpModelRoutingCron implements Runnable {

    private Client client;
    private DiscoveryNodeFilterer nodeFilter;

    public SyncUpModelRoutingCron(Client client, DiscoveryNodeFilterer nodeFilter) {
        this.client = client;
        this.nodeFilter = nodeFilter;
    }

    @Override
    public void run() {
        log.debug("sync model routing job starts");
        DiscoveryNode[] allNodes = nodeFilter.getAllNodes();
        MLSyncUpInput syncUpInput = MLSyncUpInput.builder().getLoadedModels(true).build();
        MLSyncUpNodesRequest syncUpRequest = new MLSyncUpNodesRequest(allNodes, syncUpInput);
        client.execute(MLSyncUpOnNodeAction.INSTANCE, syncUpRequest, ActionListener.wrap(r -> {
            List<MLSyncUpNodeResponse> responses = r.getNodes();
            Map<String, Set<String>> modelRoutingTable = new HashMap<>();
            for (MLSyncUpNodeResponse response : responses) {
                String nodeId = response.getNode().getId();
                String[] loadedModelIds = response.getLoadedModelIds();
                if (loadedModelIds != null && loadedModelIds.length > 0) {
                    for (String modelId : loadedModelIds) {
                        Set<String> workerNodes = modelRoutingTable.computeIfAbsent(modelId, it -> new HashSet<>());
                        workerNodes.add(nodeId);
                    }
                }
            }
            for (Map.Entry<String, Set<String>> entry : modelRoutingTable.entrySet()) {
                log.debug("will sync model routing job for model: {}: {}", entry.getKey(), entry.getValue().toArray(new String[0]));
            }
            if (modelRoutingTable.size() > 0) {
                MLSyncUpInput syncUpInput2 = MLSyncUpInput.builder().modelRoutingTable(modelRoutingTable).build();
                MLSyncUpNodesRequest syncUpRequest2 = new MLSyncUpNodesRequest(allNodes, syncUpInput2);
                client
                    .execute(
                        MLSyncUpOnNodeAction.INSTANCE,
                        syncUpRequest2,
                        ActionListener
                            .wrap(
                                re -> { log.debug("sync model routing job finished"); },
                                ex -> { log.error("Failed to sync model routing", ex); }
                            )
                    );
            }
        }, e -> { log.error("Failed to sync model routing", e); }));
    }
}
