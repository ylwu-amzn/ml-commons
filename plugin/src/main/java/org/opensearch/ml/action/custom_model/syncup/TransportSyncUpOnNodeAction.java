/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.custom_model.syncup;

import java.io.IOException;
import java.util.List;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.FailedNodeException;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.nodes.TransportNodesAction;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpInput;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpNodeRequest;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpNodeResponse;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpNodesRequest;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpNodesResponse;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpOnNodeAction;
import org.opensearch.ml.engine.algorithms.custom.CustomModelManager;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

@Log4j2
public class TransportSyncUpOnNodeAction extends
    TransportNodesAction<MLSyncUpNodesRequest, MLSyncUpNodesResponse, MLSyncUpNodeRequest, MLSyncUpNodeResponse> {
    TransportService transportService;
    CustomModelManager customModelManager;
    MLTaskManager mlTaskManager;
    MLModelManager mlModelManager;
    ClusterService clusterService;
    ThreadPool threadPool;
    Client client;
    NamedXContentRegistry xContentRegistry;

    @Inject
    public TransportSyncUpOnNodeAction(
        TransportService transportService,
        ActionFilters actionFilters,
        CustomModelManager customModelManager,
        MLTaskManager mlTaskManager,
        MLModelManager mlModelManager,
        ClusterService clusterService,
        ThreadPool threadPool,
        Client client,
        NamedXContentRegistry xContentRegistry
    ) {
        super(
            MLSyncUpOnNodeAction.NAME,
            threadPool,
            clusterService,
            transportService,
            actionFilters,
            MLSyncUpNodesRequest::new,
            MLSyncUpNodeRequest::new,
            ThreadPool.Names.MANAGEMENT,
            MLSyncUpNodeResponse.class
        );
        this.transportService = transportService;
        this.customModelManager = customModelManager;
        this.mlTaskManager = mlTaskManager;
        this.mlModelManager = mlModelManager;
        this.clusterService = clusterService;
        this.threadPool = threadPool;
        this.client = client;
        this.xContentRegistry = xContentRegistry;
    }

    @Override
    protected MLSyncUpNodesResponse newResponse(
        MLSyncUpNodesRequest nodesRequest,
        List<MLSyncUpNodeResponse> responses,
        List<FailedNodeException> failures
    ) {
        return new MLSyncUpNodesResponse(clusterService.getClusterName(), responses, failures);
    }

    @Override
    protected MLSyncUpNodeRequest newNodeRequest(MLSyncUpNodesRequest request) {
        return new MLSyncUpNodeRequest(request);
    }

    @Override
    protected MLSyncUpNodeResponse newNodeResponse(StreamInput in) throws IOException {
        return new MLSyncUpNodeResponse(in);
    }

    @Override
    protected MLSyncUpNodeResponse nodeOperation(MLSyncUpNodeRequest request) {
        return createSyncUpNodeResponse(request.getSyncUpNodesRequest());
    }

    private MLSyncUpNodeResponse createSyncUpNodeResponse(MLSyncUpNodesRequest loadModelNodesRequest) {
        MLSyncUpInput syncUpInput = loadModelNodesRequest.getSyncUpInput();
        String modelId = syncUpInput.getModelId();
        String[] addedWorkerNodes = syncUpInput.getAddedWorkerNodes();
        String[] removedWorkerNodes = syncUpInput.getRemovedWorkerNOdes();

        mlModelManager.addModelWorkerNode(modelId, addedWorkerNodes);
        mlModelManager.removeModelWorkerNode(modelId, removedWorkerNodes);

        return new MLSyncUpNodeResponse(clusterService.localNode(), "ok");
    }

}
