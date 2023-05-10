/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.undeploy;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import lombok.extern.log4j.Log4j2;
import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.client.Client;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.settings.Settings;
import org.opensearch.commons.authuser.User;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.ml.cluster.DiscoveryNodeHelper;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.MLTaskState;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.common.transport.deploy.MLDeployModelInput;
import org.opensearch.ml.common.transport.deploy.MLDeployModelNodesRequest;
import org.opensearch.ml.common.transport.deploy.MLDeployModelNodesResponse;
import org.opensearch.ml.common.transport.deploy.MLDeployModelOnNodeAction;
import org.opensearch.ml.common.transport.deploy.MLDeployModelRequest;
import org.opensearch.ml.common.transport.undeploy.MLUndeployModelsAction;
import org.opensearch.ml.common.transport.undeploy.MLUndeployModelsRequest;
import org.opensearch.ml.common.transport.undeploy.MLUndeployModelsResponse;
import org.opensearch.ml.engine.ModelHelper;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.stats.MLStats;
import org.opensearch.ml.task.MLTaskDispatcher;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.ml.utils.MLExceptionUtils;
import org.opensearch.ml.utils.RestActionUtils;
import org.opensearch.tasks.Task;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

import java.util.List;
import java.util.stream.Collectors;

import static org.opensearch.ml.common.MLTask.STATE_FIELD;
import static org.opensearch.ml.common.MLTaskState.FAILED;
import static org.opensearch.ml.settings.MLCommonsSettings.ML_COMMONS_ALLOW_CUSTOM_DEPLOYMENT_PLAN;
import static org.opensearch.ml.settings.MLCommonsSettings.ML_COMMONS_VALIDATE_BACKEND_ROLES;
import static org.opensearch.ml.task.MLTaskManager.TASK_SEMAPHORE_TIMEOUT;

;

@Log4j2
public class TransportUndeployModelsAction extends HandledTransportAction<ActionRequest, MLUndeployModelsResponse> {
    TransportService transportService;
    ModelHelper modelHelper;
    MLTaskManager mlTaskManager;
    ClusterService clusterService;
    ThreadPool threadPool;
    Client client;
    NamedXContentRegistry xContentRegistry;
    DiscoveryNodeHelper nodeFilter;
    MLTaskDispatcher mlTaskDispatcher;
    MLModelManager mlModelManager;
    MLStats mlStats;

    private volatile boolean allowCustomDeploymentPlan;
    private volatile boolean filterByEnabled;

    @Inject
    public TransportUndeployModelsAction(
        TransportService transportService,
        ActionFilters actionFilters,
        ModelHelper modelHelper,
        MLTaskManager mlTaskManager,
        ClusterService clusterService,
        ThreadPool threadPool,
        Client client,
        NamedXContentRegistry xContentRegistry,
        DiscoveryNodeHelper nodeFilter,
        MLTaskDispatcher mlTaskDispatcher,
        MLModelManager mlModelManager,
        MLStats mlStats,
        Settings settings
    ) {
        super(MLUndeployModelsAction.NAME, transportService, actionFilters, MLDeployModelRequest::new);
        this.transportService = transportService;
        this.modelHelper = modelHelper;
        this.mlTaskManager = mlTaskManager;
        this.clusterService = clusterService;
        this.threadPool = threadPool;
        this.client = client;
        this.xContentRegistry = xContentRegistry;
        this.nodeFilter = nodeFilter;
        this.mlTaskDispatcher = mlTaskDispatcher;
        this.mlModelManager = mlModelManager;
        this.mlStats = mlStats;
        allowCustomDeploymentPlan = ML_COMMONS_ALLOW_CUSTOM_DEPLOYMENT_PLAN.get(settings);
        filterByEnabled = ML_COMMONS_VALIDATE_BACKEND_ROLES.get(settings);
        clusterService
            .getClusterSettings()
            .addSettingsUpdateConsumer(ML_COMMONS_ALLOW_CUSTOM_DEPLOYMENT_PLAN, it -> allowCustomDeploymentPlan = it);
        clusterService.getClusterSettings().addSettingsUpdateConsumer(ML_COMMONS_VALIDATE_BACKEND_ROLES, it -> filterByEnabled = it);

    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<MLUndeployModelsResponse> listener) {
        //MLDeployModelRequest deployModelRequest = MLDeployModelRequest.fromActionRequest(request);
        //String modelId = deployModelRequest.getModelId();
        MLUndeployModelsRequest undeployModelsRequest = MLUndeployModelsRequest.fromActionRequest(request);

        // You can get user information in this transport action.
        User user = RestActionUtils.getUserContext(client);
        String[] excludes = new String[] { MLModel.MODEL_CONTENT_FIELD, MLModel.OLD_MODEL_CONTENT_FIELD };

        // TODO: then you can send out request to undeploy models
        //client.execute(MLUndeployModelAction.INSTANCE, mlUndeployModelNodesRequest, ActionListener.wrap());
        listener.onFailure(new MLException("test error"));
    }

    @VisibleForTesting
    void updateModelDeployStatusAndTriggerOnNodesAction(
        String modelId,
        String taskId,
        MLModel mlModel,
        String localNodeId,
        MLTask mlTask,
        List<DiscoveryNode> eligibleNodes,
        boolean deployToAllNodes
    ) {
        MLDeployModelInput deployModelInput = new MLDeployModelInput(
            modelId,
            taskId,
            mlModel.getModelContentHash(),
            eligibleNodes.size(),
            localNodeId,
            mlTask
        );
        MLDeployModelNodesRequest deployModelRequest = new MLDeployModelNodesRequest(
            eligibleNodes.toArray(new DiscoveryNode[0]),
            deployModelInput
        );
        ActionListener<MLDeployModelNodesResponse> actionListener = ActionListener.wrap(r -> {
            if (mlTaskManager.contains(taskId)) {
                mlTaskManager.updateMLTask(taskId, ImmutableMap.of(STATE_FIELD, MLTaskState.RUNNING), TASK_SEMAPHORE_TIMEOUT, false);
            }
        }, e -> {
            log.error("Failed to deploy model " + modelId, e);
            mlTaskManager
                .updateMLTask(
                    taskId,
                    ImmutableMap.of(MLTask.ERROR_FIELD, MLExceptionUtils.getRootCauseMessage(e), STATE_FIELD, FAILED),
                    TASK_SEMAPHORE_TIMEOUT,
                    true
                );
            mlModelManager.updateModel(modelId, ImmutableMap.of(MLModel.MODEL_STATE_FIELD, MLModelState.DEPLOY_FAILED));
        });

        List<String> workerNodes = eligibleNodes.stream().map(n -> n.getId()).collect(Collectors.toList());
        mlModelManager
            .updateModel(
                modelId,
                ImmutableMap
                    .of(
                        MLModel.MODEL_STATE_FIELD,
                        MLModelState.DEPLOYING,
                        MLModel.PLANNING_WORKER_NODE_COUNT_FIELD,
                        eligibleNodes.size(),
                        MLModel.PLANNING_WORKER_NODES_FIELD,
                        workerNodes,
                        MLModel.DEPLOY_TO_ALL_NODES_FIELD,
                        deployToAllNodes
                    ),
                ActionListener
                    .wrap(
                        r -> client.execute(MLDeployModelOnNodeAction.INSTANCE, deployModelRequest, actionListener),
                        actionListener::onFailure
                    )
            );
    }

}
