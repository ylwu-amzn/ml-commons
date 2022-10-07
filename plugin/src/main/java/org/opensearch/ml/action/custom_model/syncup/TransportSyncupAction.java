/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.custom_model.syncup;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpAction;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpInput;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpNodesRequest;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpOnNodeAction;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpRequest;
import org.opensearch.ml.common.transport.custom_model.sync.MLSyncUpResponse;
import org.opensearch.ml.engine.algorithms.custom.CustomModelManager;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.task.MLTaskDispatcher;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.tasks.Task;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

@Log4j2
public class TransportSyncupAction extends HandledTransportAction<ActionRequest, MLSyncUpResponse> {
    TransportService transportService;
    CustomModelManager customModelManager;
    MLTaskManager mlTaskManager;
    ClusterService clusterService;
    ThreadPool threadPool;
    Client client;
    NamedXContentRegistry xContentRegistry;
    MLTaskDispatcher mlTaskDispatcher;
    MLModelManager mlModelManager;

    @Inject
    public TransportSyncupAction(
        TransportService transportService,
        ActionFilters actionFilters,
        CustomModelManager customModelManager,
        MLTaskManager mlTaskManager,
        ClusterService clusterService,
        ThreadPool threadPool,
        Client client,
        NamedXContentRegistry xContentRegistry,
        MLTaskDispatcher mlTaskDispatcher,
        MLModelManager mlModelManager
    ) {
        super(MLSyncUpAction.NAME, transportService, actionFilters, MLSyncUpRequest::new);
        this.transportService = transportService;
        this.customModelManager = customModelManager;
        this.mlTaskManager = mlTaskManager;
        this.clusterService = clusterService;
        this.threadPool = threadPool;
        this.client = client;
        this.xContentRegistry = xContentRegistry;
        this.mlTaskDispatcher = mlTaskDispatcher;
        this.mlModelManager = mlModelManager;
    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<MLSyncUpResponse> listener) {
        MLSyncUpRequest deployModelRequest = MLSyncUpRequest.fromActionRequest(request);
        String modelId = deployModelRequest.getModelId();
        String[] addedWorkerNodes = deployModelRequest.getAddedWorkerNodes();
        String[] removedWorkerNodes = deployModelRequest.getRemovedWorkerNodes();
        try {
            String[] nodeIds = mlTaskDispatcher.getAllNodes();
            MLSyncUpInput syncUpInput = MLSyncUpInput
                .builder()
                .modelId(modelId)
                .addedWorkerNodes(addedWorkerNodes)
                .removedWorkerNodes(removedWorkerNodes)
                .build();
            MLSyncUpNodesRequest syncUpRequest = new MLSyncUpNodesRequest(nodeIds, syncUpInput);
            client
                .execute(
                    MLSyncUpOnNodeAction.INSTANCE,
                    syncUpRequest,
                    ActionListener.wrap(r -> { listener.onResponse(new MLSyncUpResponse("ok")); }, e -> {
                        e.printStackTrace();
                        listener.onFailure(e);
                    })
                );
        } catch (Exception e) {
            log.error("Failed to download custom model " + modelId, e);
            listener.onFailure(e);
        }
    }

}
