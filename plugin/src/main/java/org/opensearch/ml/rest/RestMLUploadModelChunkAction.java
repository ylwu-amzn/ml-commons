/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import org.opensearch.client.node.NodeClient;
import org.opensearch.ml.common.transport.model.upload_chunk.MLUploadChunkInput;
import org.opensearch.ml.common.transport.model.upload_chunk.MLUploadModelChunkAction;
import org.opensearch.ml.common.transport.model.upload_chunk.MLUploadModelChunkRequest;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;

public class RestMLUploadModelChunkAction extends BaseRestHandler {
    private static final String ML_UPLOAD_MODEL_CHUNK_ACTION = "ml_upload_model_chunk_action";

    /**
     * Constructor
     */
    public RestMLUploadModelChunkAction() {}

    @Override
    public String getName() {
        return ML_UPLOAD_MODEL_CHUNK_ACTION;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList
            .of(
                new Route(
                    RestRequest.Method.POST,
                    String.format(Locale.ROOT, "%s/models/{%s}/chunk/{%s}", ML_BASE_URI, "model_id", "chunk_number")
                )
            );
    }

    @Override
    public RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        MLUploadModelChunkRequest mlUploadModelRequest = getRequest(request);
        return channel -> client.execute(MLUploadModelChunkAction.INSTANCE, mlUploadModelRequest, new RestToXContentListener<>(channel));
    }

    /**
     * Creates a MLTrainingTaskRequest from a RestRequest
     *
     * @param request RestRequest
     * @return MLUploadModelChunkRequest
     */
    @VisibleForTesting
    MLUploadModelChunkRequest getRequest(RestRequest request) throws IOException {
        final var modelId = request.param("model_id");
        String chunk_number = request.param("chunk_number");
        byte[] content = request.content().streamInput().readAllBytes();
        MLUploadChunkInput mlInput = new MLUploadChunkInput(modelId, Integer.parseInt(chunk_number), content);
        return new MLUploadModelChunkRequest(mlInput);
    }
}
