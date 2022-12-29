/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.upload;

import static org.opensearch.ml.common.MLTask.STATE_FIELD;
import static org.opensearch.ml.common.MLTaskState.FAILED;
import static org.opensearch.ml.settings.MLCommonsSettings.ML_COMMONS_TRUSTED_URL_REGEX;
import static org.opensearch.ml.task.MLTaskManager.TASK_SEMAPHORE_TIMEOUT;
import static org.opensearch.ml.utils.RestActionUtils.handleException;

import java.time.Instant;
import java.util.Arrays;
import java.util.regex.Pattern;

import com.google.common.collect.ImmutableMap;
import lombok.extern.log4j.Log4j2;

import org.apache.commons.lang3.exception.ExceptionUtils;
import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionListenerResponseHandler;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.ml.cluster.DiscoveryNodeHelper;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.MLTaskState;
import org.opensearch.ml.common.MLTaskType;
import org.opensearch.ml.common.exception.MLLimitExceededException;
import org.opensearch.ml.common.transport.forward.MLForwardAction;
import org.opensearch.ml.common.transport.forward.MLForwardInput;
import org.opensearch.ml.common.transport.forward.MLForwardRequest;
import org.opensearch.ml.common.transport.forward.MLForwardRequestType;
import org.opensearch.ml.common.transport.forward.MLForwardResponse;
import org.opensearch.ml.common.transport.upload.MLUploadInput;
import org.opensearch.ml.common.transport.upload.MLUploadModelAction;
import org.opensearch.ml.common.transport.upload.MLUploadModelRequest;
import org.opensearch.ml.common.transport.upload.UploadModelResponse;
import org.opensearch.ml.engine.ModelHelper;
import org.opensearch.ml.indices.MLIndicesHandler;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.stats.MLNodeLevelStat;
import org.opensearch.ml.stats.MLStats;
import org.opensearch.ml.task.MLTaskDispatcher;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.ml.utils.MLExceptionUtils;
import org.opensearch.tasks.Task;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

@Log4j2
public class TransportUploadModelAction extends HandledTransportAction<ActionRequest, UploadModelResponse> {
    TransportService transportService;
    ModelHelper modelHelper;
    MLIndicesHandler mlIndicesHandler;
    MLModelManager mlModelManager;
    MLTaskManager mlTaskManager;
    ClusterService clusterService;
    ThreadPool threadPool;
    Client client;
    DiscoveryNodeHelper nodeFilter;
    MLTaskDispatcher mlTaskDispatcher;
    MLStats mlStats;
    String trustedUrlRegex;

    @Inject
    public TransportUploadModelAction(
        TransportService transportService,
        ActionFilters actionFilters,
        ModelHelper modelHelper,
        MLIndicesHandler mlIndicesHandler,
        MLModelManager mlModelManager,
        MLTaskManager mlTaskManager,
        ClusterService clusterService,
        Settings settings,
        ThreadPool threadPool,
        Client client,
        DiscoveryNodeHelper nodeFilter,
        MLTaskDispatcher mlTaskDispatcher,
        MLStats mlStats
    ) {
        super(MLUploadModelAction.NAME, transportService, actionFilters, MLUploadModelRequest::new);
        this.transportService = transportService;
        this.modelHelper = modelHelper;
        this.mlIndicesHandler = mlIndicesHandler;
        this.mlModelManager = mlModelManager;
        this.mlTaskManager = mlTaskManager;
        this.clusterService = clusterService;
        this.threadPool = threadPool;
        this.client = client;
        this.nodeFilter = nodeFilter;
        this.mlTaskDispatcher = mlTaskDispatcher;
        this.mlStats = mlStats;

        trustedUrlRegex = ML_COMMONS_TRUSTED_URL_REGEX.get(settings);
        clusterService.getClusterSettings().addSettingsUpdateConsumer(ML_COMMONS_TRUSTED_URL_REGEX, it -> trustedUrlRegex = it);
    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<UploadModelResponse> listener) {
        MLUploadModelRequest uploadModelRequest = MLUploadModelRequest.fromActionRequest(request);
        MLUploadInput mlUploadInput = uploadModelRequest.getMlUploadInput();
        Pattern pattern = Pattern.compile(trustedUrlRegex);
        String url = mlUploadInput.getUrl();
        if (url != null) {
            boolean validUrl = pattern.matcher(url).find();
            if (!validUrl) {
                throw new IllegalArgumentException("URL can't match trusted url regex");
            }
        }
        // mlStats.getStat(MLNodeLevelStat.ML_NODE_EXECUTING_TASK_COUNT).increment();
        mlStats.getStat(MLNodeLevelStat.ML_NODE_TOTAL_REQUEST_COUNT).increment();
        // //TODO: track executing task; track upload failures
        // mlStats.createCounterStatIfAbsent(FunctionName.TEXT_EMBEDDING, ActionName.UPLOAD,
        // MLActionLevelStat.ML_ACTION_REQUEST_COUNT).increment();
        MLTask mlTask = MLTask
            .builder()
            .async(true)
            .taskType(MLTaskType.UPLOAD_MODEL)
            .functionName(mlUploadInput.getFunctionName())
            .createTime(Instant.now())
            .lastUpdateTime(Instant.now())
            .state(MLTaskState.CREATED)
            .workerNode(clusterService.localNode().getId())
            .build();

        mlTaskDispatcher.dispatch(ActionListener.wrap(node -> {
            String nodeId = node.getId();
            mlTask.setWorkerNode(nodeId);

            mlTaskManager.createMLTask(mlTask, ActionListener.wrap(response -> {
                String taskId = response.getId();
                mlTask.setTaskId(taskId);
                listener.onResponse(new UploadModelResponse(taskId, MLTaskState.CREATED.name()));

                ActionListener<MLForwardResponse> forwardActionListener = ActionListener
                        .wrap(
                                res -> {
                                    log.info("---------- Upload model response: " + res);
                                    boolean removeTaskCache = !clusterService.localNode().getId().equals(nodeId);
                                    if (removeTaskCache) {
                                        log.info("----------------------------- remove task from cache " + taskId);
                                        mlTaskManager.remove(taskId);
                                    }
                                    },
                                ex -> {
                                    log.error("---------- Failed to upload model", ex);
                                    mlTaskManager
                                            .updateMLTask(
                                                    taskId,
                                                    ImmutableMap.of(MLTask.ERROR_FIELD, MLExceptionUtils.getRootCauseMessage(ex), STATE_FIELD, FAILED),
                                                    TASK_SEMAPHORE_TIMEOUT,
                                                    true
                                            );
                                }
                        );
                try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                    mlTaskManager.add(mlTask, Arrays.asList(nodeId));
                    MLForwardInput forwardInput = MLForwardInput
                            .builder()
                            .requestType(MLForwardRequestType.UPLOAD_MODEL)
                            .uploadInput(mlUploadInput)
                            .mlTask(mlTask)
                            .build();
                    MLForwardRequest forwardRequest = new MLForwardRequest(forwardInput);
                    transportService
                            .sendRequest(
                                    node,
                                    MLForwardAction.NAME,
                                    forwardRequest,
                                    new ActionListenerResponseHandler<>(forwardActionListener, MLForwardResponse::new)
                            );
                } catch (Exception e) {
                    forwardActionListener.onFailure(e);
                }

//                if (clusterService.localNode().getId().equals(nodeId)) {
//                    mlModelManager.uploadMLModel(mlUploadInput, mlTask);
//                } else {
//                    MLForwardInput forwardInput = MLForwardInput
//                        .builder()
//                        .requestType(MLForwardRequestType.UPLOAD_MODEL)
//                        .uploadInput(mlUploadInput)
//                        .mlTask(mlTask)
//                        .build();
//                    MLForwardRequest forwardRequest = new MLForwardRequest(forwardInput);
//                    ActionListener<MLForwardResponse> myListener = ActionListener
//                        .wrap(
//                            res -> { log.debug("Response from model node: " + res); },
//                            ex -> { log.error("Failure from model node", ex); }
//                        );
//                    try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
//                        transportService
//                            .sendRequest(
//                                node,
//                                MLForwardAction.NAME,
//                                forwardRequest,
//                                new ActionListenerResponseHandler<>(myListener, MLForwardResponse::new)
//                            );
//                    }
//                }
            }, e -> {
                handleException(listener, e, "Failed to upload model");
            }));
        }, e -> {
            handleException(listener, e, "Failed to upload model");
        }));

    }

//    private static void handleException(ActionListener<?> listener, Exception e) {
//        if (e instanceof MLLimitExceededException) {
//            log.warn(e.getMessage());
//        } else {
//            log.error("Failed to upload model ", e);
//        }
//        listener.onFailure(e);
//    }
}
