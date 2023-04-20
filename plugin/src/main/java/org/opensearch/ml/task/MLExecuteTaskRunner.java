/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.task;

import static org.opensearch.ml.plugin.MachineLearningPlugin.EXECUTE_THREAD_POOL;
import static org.opensearch.ml.settings.MLCommonsSettings.ML_COMMONS_ENABLE_INHOUSE_PYTHON_MODEL;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionListenerResponseHandler;
import org.opensearch.client.Client;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.ml.breaker.MLCircuitBreakerService;
import org.opensearch.ml.cluster.DiscoveryNodeHelper;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.input.Input;
import org.opensearch.ml.common.output.Output;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskAction;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskRequest;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskResponse;
import org.opensearch.ml.engine.MLEngine;
import org.opensearch.ml.indices.MLInputDatasetHandler;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.stats.ActionName;
import org.opensearch.ml.stats.MLActionLevelStat;
import org.opensearch.ml.stats.MLNodeLevelStat;
import org.opensearch.ml.stats.MLStats;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportResponseHandler;
import org.opensearch.transport.TransportService;

/**
 * MLExecuteTaskRunner is responsible for running execute tasks.
 */
@Log4j2
public class MLExecuteTaskRunner extends MLTaskRunner<MLExecuteTaskRequest, MLExecuteTaskResponse> {
    private final ThreadPool threadPool;
    private final ClusterService clusterService;
    private final Client client;
    private final MLInputDatasetHandler mlInputDatasetHandler;
    protected final DiscoveryNodeHelper nodeHelper;
    private final MLEngine mlEngine;
    private volatile Boolean isPythonModelEnabled;
    private final MLModelManager mlModelManager;

    public MLExecuteTaskRunner(
        ThreadPool threadPool,
        ClusterService clusterService,
        Client client,
        MLTaskManager mlTaskManager,
        MLStats mlStats,
        MLInputDatasetHandler mlInputDatasetHandler,
        MLTaskDispatcher mlTaskDispatcher,
        MLCircuitBreakerService mlCircuitBreakerService,
        DiscoveryNodeHelper nodeHelper,
        MLEngine mlEngine,
        MLModelManager mlModelManager
    ) {
        super(mlTaskManager, mlStats, nodeHelper, mlTaskDispatcher, mlCircuitBreakerService, clusterService);
        this.threadPool = threadPool;
        this.clusterService = clusterService;
        this.client = client;
        this.mlInputDatasetHandler = mlInputDatasetHandler;
        this.nodeHelper = nodeHelper;
        this.mlEngine = mlEngine;
        isPythonModelEnabled = ML_COMMONS_ENABLE_INHOUSE_PYTHON_MODEL.get(this.clusterService.getSettings());
        this.clusterService
            .getClusterSettings()
            .addSettingsUpdateConsumer(ML_COMMONS_ENABLE_INHOUSE_PYTHON_MODEL, it -> isPythonModelEnabled = it);
        this.mlModelManager = mlModelManager;
    }

    @Override
    protected String getTransportActionName() {
        return MLExecuteTaskAction.NAME;
    }

    @Override
    protected TransportResponseHandler<MLExecuteTaskResponse> getResponseHandler(ActionListener<MLExecuteTaskResponse> listener) {
        return new ActionListenerResponseHandler<>(listener, MLExecuteTaskResponse::new);
    }

    @Override
    public void dispatchTask(MLExecuteTaskRequest request, TransportService transportService, ActionListener<MLExecuteTaskResponse> listener) {
        request.getFunctionName();
        String modelId = request.getFunctionName().name();
        try {
            ActionListener<DiscoveryNode> actionListener = ActionListener.wrap(node -> {
                if (clusterService.localNode().getId().equals(node.getId())) {
                    log.debug("Execute ML predict request {} locally on node {}", request.getRequestID(), node.getId());
                    executeTask(request, listener);
                } else {
                    log.debug("Execute ML predict request {} remotely on node {}", request.getRequestID(), node.getId());
                    request.setDispatchTask(false);
                    transportService.sendRequest(node, getTransportActionName(), request, getResponseHandler(listener));
                }
            }, e -> { listener.onFailure(e); });
            String[] workerNodes = mlModelManager.getWorkerNodes(modelId, true);
            if (workerNodes == null || workerNodes.length == 0) {
                if (request.getFunctionName() == FunctionName.METRICS_CORRELATION) {
                    listener.onFailure(new IllegalArgumentException("model not deployed"));
                    return;
                } else {
                    workerNodes = nodeHelper.getEligibleNodeIds();
                }
            }
            mlTaskDispatcher.dispatchPredictTask(workerNodes, actionListener);
        } catch (Exception e) {
            log.error("Failed to predict model " + modelId, e);
            listener.onFailure(e);
        }
    }

    /**
     * Execute algorithm and return result.
     * @param request MLExecuteTaskRequest
     * @param listener Action listener
     */
    @Override
    protected void executeTask(MLExecuteTaskRequest request, ActionListener<MLExecuteTaskResponse> listener) {
        threadPool.executor(EXECUTE_THREAD_POOL).execute(() -> {
            try {
                mlStats.getStat(MLNodeLevelStat.ML_NODE_EXECUTING_TASK_COUNT).increment();
                mlStats.getStat(MLNodeLevelStat.ML_NODE_TOTAL_REQUEST_COUNT).increment();
                mlStats
                    .createCounterStatIfAbsent(request.getFunctionName(), ActionName.EXECUTE, MLActionLevelStat.ML_ACTION_REQUEST_COUNT)
                    .increment();

                // ActionListener<MLExecuteTaskResponse> wrappedListener = ActionListener.runBefore(listener, )
                Input input = request.getInput();
                FunctionName functionName = request.getFunctionName();
                if (FunctionName.METRICS_CORRELATION.equals(functionName)) {
                    if (!isPythonModelEnabled) {
                        Exception exception = new IllegalArgumentException("This algorithm is not enabled from settings");
                        listener.onFailure(exception);
                        return;
                    }
                }
                Output output = mlEngine.execute(input);
                MLExecuteTaskResponse response = new MLExecuteTaskResponse(functionName, output);
                listener.onResponse(response);
            } catch (Exception e) {
                mlStats
                    .createCounterStatIfAbsent(request.getFunctionName(), ActionName.EXECUTE, MLActionLevelStat.ML_ACTION_FAILURE_COUNT)
                    .increment();
                listener.onFailure(e);
            } finally {
                mlStats.getStat(MLNodeLevelStat.ML_NODE_EXECUTING_TASK_COUNT).decrement();
            }
        });
    }

}
