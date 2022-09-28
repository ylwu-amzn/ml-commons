/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.custom_model.forward;

import lombok.extern.log4j.Log4j2;
import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.ml.action.custom_model.upload.MLModelUploader;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.model.MLModelConfig;
import org.opensearch.ml.common.model.MLModelFormat;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardAction;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardInput;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardRequest;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardRequestType;
import org.opensearch.ml.common.transport.custom_model.forward.MLForwardResponse;
import org.opensearch.ml.common.transport.custom_model.upload.MLUploadInput;
import org.opensearch.ml.engine.algorithms.custom.CustomModelManager;
import org.opensearch.ml.indices.MLIndicesHandler;
import org.opensearch.ml.task.MLTaskDispatcher;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.tasks.Task;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;

@Log4j2
public class TransportForwardAction extends HandledTransportAction<ActionRequest, MLForwardResponse> {
    private MLModelUploader mlModelUploader;

    @Inject
    public TransportForwardAction(
        TransportService transportService,
        ActionFilters actionFilters,
        MLModelUploader mlModelUploader
    ) {
        super(MLForwardAction.NAME, transportService, actionFilters, MLForwardRequest::new);
        this.mlModelUploader = mlModelUploader;
    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<MLForwardResponse> listener) {
        MLForwardRequest mlForwardRequest = MLForwardRequest.fromActionRequest(request);
        MLForwardInput forwardInput = mlForwardRequest.getForwardInput();
        String modelName = forwardInput.getName();
        Integer version = forwardInput.getVersion();
        String url = forwardInput.getUrl();
        MLModelFormat modelFormat = forwardInput.getModelFormat();
        MLModelConfig modelConfig = forwardInput.getModelConfig();
        MLTask mlTask = forwardInput.getMlTask();
        MLForwardRequestType requestType = forwardInput.getRequestType();
        log.debug("Received ML forward request: {}", requestType);
        try {
            switch (requestType) {
                case UPLOAD_MODEL:
                    mlModelUploader.uploadModel(new MLUploadInput(modelName, version, url, modelFormat, modelConfig), mlTask);
                    listener.onResponse(new MLForwardResponse("ok", null));
                    break;
                default:
                    break;
            }

        } catch (Exception e) {
            log.error("Failed to execute ML forward action for " + modelName, e);
            listener.onFailure(e);
        }
    }
}
