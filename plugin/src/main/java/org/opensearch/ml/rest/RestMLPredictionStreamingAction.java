/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.core.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;
import static org.opensearch.ml.utils.MLExceptionUtils.BATCH_INFERENCE_DISABLED_ERR_MSG;
import static org.opensearch.ml.utils.MLExceptionUtils.LOCAL_MODEL_DISABLED_ERR_MSG;
import static org.opensearch.ml.utils.MLExceptionUtils.REMOTE_INFERENCE_DISABLED_ERR_MSG;
import static org.opensearch.ml.utils.MLExceptionUtils.STREAM_DISABLED_ERR_MSG;
import static org.opensearch.ml.utils.RestActionUtils.PARAMETER_ALGORITHM;
import static org.opensearch.ml.utils.RestActionUtils.PARAMETER_MODEL_ID;
import static org.opensearch.ml.utils.RestActionUtils.getActionTypeFromRestRequest;
import static org.opensearch.ml.utils.RestActionUtils.getParameterId;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.lease.Releasable;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.support.XContentHttpChunk;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.transport.TransportResponse;
import org.opensearch.core.xcontent.MediaType;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.http.HttpChunk;
import org.opensearch.ml.action.prediction.TransportPredictionTaskAction;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.connector.ConnectorAction.ActionType;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.remote.RemoteInferenceMLInput;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.plugin.MachineLearningPlugin;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.BytesRestResponse;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.StreamingRestChannel;
import org.opensearch.tasks.Task;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.StreamTransportResponseHandler;
import org.opensearch.transport.StreamTransportService;
import org.opensearch.transport.TransportException;
import org.opensearch.transport.TransportRequest;
import org.opensearch.transport.TransportRequestOptions;
import org.opensearch.transport.client.node.NodeClient;
import org.opensearch.transport.stream.StreamTransportResponse;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;

import lombok.extern.log4j.Log4j2;
import reactor.core.publisher.Flux;

@Log4j2
public class RestMLPredictionStreamingAction extends BaseRestHandler {

    public class StreamPredictActionListener<Response extends TransportResponse, Request extends TransportRequest>
        implements
            ActionListener<Response> {

        private final StreamingRestChannel restChannel;
        private final String actionName;
        private final Request request;
        private final NodeClient client;

        // Constructor for REST layer
        public StreamPredictActionListener(StreamingRestChannel restChannel, String actionName, Request request, NodeClient client) {
            this.restChannel = restChannel;
            this.actionName = actionName;
            this.request = request;
            this.client = client;
        }

//        // Add getter method
//        public StreamingRestChannel getRestChannel() {
//            return restChannel;
//        }

        public void onStreamResponse(Response response, boolean isLastBatch) {
            log.info("REST onStreamResponse received, isLastBatch: {}", isLastBatch);

            MLTaskResponse mlResponse = (MLTaskResponse) response;
            String content = extractContent(mlResponse);
            boolean isLast = isLastChunk(mlResponse);
            log.info("Extracted content: '{}', isLast from content: {}", content, isLast);

            try {
                restChannel.sendChunk(convertToHttpChunk(mlResponse));
                log.info("Sent chunk with content: '{}', isLast: {}", content, isLast);
            } catch (IOException e) {
                log.error("Failed to send chunk", e);
            }

            // Send final marker when stream is complete
            if (isLast) {
                log.info("Stream completed - sending final marker");
                restChannel.sendChunk(XContentHttpChunk.last());
            }
        }

        private String extractContent(MLTaskResponse response) {
            try {
                ModelTensorOutput output = (ModelTensorOutput) response.getOutput();
                if (output != null && !output.getMlModelOutputs().isEmpty()) {
                    ModelTensors modelTensors = output.getMlModelOutputs().get(0);
                    if (!modelTensors.getMlModelTensors().isEmpty()) {
                        Map<String, ?> dataMap = modelTensors.getMlModelTensors().get(0).getDataAsMap();
                        if (dataMap.containsKey("content")) {
                            return (String) dataMap.get("content");
                        }
                    }
                }
            } catch (Exception e) {
                log.error("Failed to extract content", e);
            }
            return "";
        }

        private boolean isLastChunk(MLTaskResponse response) {
            try {
                ModelTensorOutput output = (ModelTensorOutput) response.getOutput();
                if (output != null && !output.getMlModelOutputs().isEmpty()) {
                    ModelTensors modelTensors = output.getMlModelOutputs().get(0);
                    if (!modelTensors.getMlModelTensors().isEmpty()) {
                        Map<String, ?> dataMap = modelTensors.getMlModelTensors().get(0).getDataAsMap();
                        if (dataMap.containsKey("is_last")) {
                            return Boolean.TRUE.equals(dataMap.get("is_last"));
                        }
                    }
                }
            } catch (Exception e) {
                log.error("Failed to check is_last", e);
            }
            return false;
        }

        @Override
        public final void onResponse(Response response) {
            MLTaskResponse mlResponse = (MLTaskResponse) response;
            boolean isLastBatch = isLastChunk(mlResponse);
            log.info("islastBatch from content is {}", isLastBatch);
            onStreamResponse(response, isLastBatch);
        }

        @Override
        public void onFailure(Exception e) {
            throw new MLException("Got an exception in MLPredictionTaskAction.", e);
        }
    }

    private static final String ML_PREDICTION_ACTION = "ml_prediction_streaming_action";

    private MLModelManager modelManager;

    private MLFeatureEnabledSetting mlFeatureEnabledSetting;

    private ClusterService clusterService;

    /**
     * Constructor
     */
    public RestMLPredictionStreamingAction(
        MLModelManager modelManager,
        MLFeatureEnabledSetting mlFeatureEnabledSetting,
        ClusterService clusterService
    ) {
        this.modelManager = modelManager;
        this.mlFeatureEnabledSetting = mlFeatureEnabledSetting;
        this.clusterService = clusterService;
    }

    @Override
    public String getName() {
        return ML_PREDICTION_ACTION;
    }

    @Override
    public List<Route> routes() {
        return ImmutableList
            .of(
                new Route(
                    RestRequest.Method.POST,
                    String.format(Locale.ROOT, "%s/models/{%s}/_predict/stream", ML_BASE_URI, PARAMETER_MODEL_ID)
                ),
                new Route(
                    RestRequest.Method.POST,
                    String.format(Locale.ROOT, "%s/models/{%s}/_batch_predict/stream", ML_BASE_URI, PARAMETER_MODEL_ID)
                )
            );
    }

    @Override
    public RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        if (!mlFeatureEnabledSetting.isStreamEnabled()) {
            throw new IllegalStateException(STREAM_DISABLED_ERR_MSG);
        }
        String algorithm = request.param(PARAMETER_ALGORITHM);
        String modelId = getParameterId(request, PARAMETER_MODEL_ID);
        Optional<FunctionName> functionName = modelManager.getOptionalModelFunctionName(modelId);

        if (algorithm == null && functionName.isPresent()) {
            algorithm = functionName.get().name();
        }

        final StreamingRestChannelConsumer consumer = (channel) -> {
            Map<String, List<String>> headers = Map
                .of(
                    "Content-Type",
                    List.of("text/event-stream"),
                    "Cache-Control",
                    List.of("no-cache"),
                    "Connection",
                    List.of("keep-alive")
                );
            channel.prepareResponse(RestStatus.OK, headers);

            Flux
                .from(channel)
                .ofType(HttpChunk.class)
                .take(1)
                .map(HttpChunk::content)
                .doOnNext(bytesReference -> {
                    try {
                        MLPredictionTaskRequest taskRequest = getRequest(modelId, FunctionName.REMOTE.name(), request, bytesReference);
                        ThreadContext threadContext = client.threadPool().getThreadContext();
                        if (threadContext.getPersistent("ml.streaming.rest.channel") == null) {
                            threadContext.putPersistent("ml.streaming.rest.channel", channel);
                        }

                        StreamTransportResponseHandler<MLTaskResponse> handler = new StreamTransportResponseHandler<MLTaskResponse>() {
                            @Override
                            public void handleStreamResponse(StreamTransportResponse<MLTaskResponse> streamResponse) {
                                try {
                                    MLTaskResponse response;
                                    int count = 0;
                                    while ((response = streamResponse.nextResponse()) != null) {
                                        log.info("Received response: {}, count {}", response, count);
                                        channel.sendChunk(convertToHttpChunk(response));
                                        count++;
                                    }
                                    channel.sendChunk(XContentHttpChunk.last());
                                    streamResponse.close();
                                } catch (Exception e) {
                                    streamResponse.cancel("Test error", e);
                                }
                            }

                            @Override
                            public void handleException(TransportException exp) {
                                // fail("Transport exception: " + exp.getMessage());
                            }

                            @Override
                            public String executor() {
                                return ThreadPool.Names.SAME;
                            }

                            @Override
                            public MLTaskResponse read(StreamInput in) throws IOException {
                                return new MLTaskResponse(in);
                            }
                        };

                        TransportPredictionTaskAction.streamTransportService
                            .sendRequest(
                                clusterService.localNode(),
                                MLPredictionTaskAction.NAME,
                                taskRequest,
                                TransportRequestOptions.builder().withType(TransportRequestOptions.Type.STREAM).build(),
                                handler
                            );
                    } catch (IOException e) {
                        throw new MLException("Got an exception in flux.", e);
                    }
                })
                .onErrorComplete(ex -> {
                    if (ex instanceof Error) {
                        log.info("Got an error in flux");
                        return false;
                    }
                    try {
                        channel.sendResponse(new BytesRestResponse(channel, (Exception) ex));
                        return true;
                    } catch (final IOException e) {
                        throw new UncheckedIOException(e);
                    }
                })
                .subscribe();
        };

        return channel -> {
            if (channel instanceof StreamingRestChannel) {
                consumer.accept((StreamingRestChannel) channel);
            } else {
                final ActionRequestValidationException validationError = new ActionRequestValidationException();
                validationError.addValidationError("Unable to initiate request / response streaming over non-streaming channel");
                channel.sendResponse(new BytesRestResponse(channel, validationError));
            }
        };
    }

    @Override
    public boolean supportsContentStream() {
        return true;
    }

    @Override
    public boolean supportsStreaming() {
        return true;
    }

    @Override
    public boolean allowsUnsafeBuffers() {
        return true;
    }

    /**
     * Creates a MLPredictionTaskRequest from a RestRequest
     *
     * @param request RestRequest
     * @return MLPredictionTaskRequest
     */
    @VisibleForTesting
    MLPredictionTaskRequest getRequest(String modelId, String algorithm, RestRequest request, BytesReference content) throws IOException {
        ActionType actionType = ActionType.from(getActionTypeFromRestRequest(request));
        if (FunctionName.REMOTE.name().equals(algorithm) && !mlFeatureEnabledSetting.isRemoteInferenceEnabled()) {
            throw new IllegalStateException(REMOTE_INFERENCE_DISABLED_ERR_MSG);
        } else if (FunctionName.isDLModel(FunctionName.from(algorithm.toUpperCase(Locale.ROOT)))
            && !mlFeatureEnabledSetting.isLocalModelEnabled()) {
            throw new IllegalStateException(LOCAL_MODEL_DISABLED_ERR_MSG);
        } else if (ActionType.BATCH_PREDICT == actionType && !mlFeatureEnabledSetting.isOfflineBatchInferenceEnabled()) {
            throw new IllegalStateException(BATCH_INFERENCE_DISABLED_ERR_MSG);
        } else if (!ActionType.isValidActionInModelPrediction(actionType)) {
            throw new IllegalArgumentException("Wrong action type in the rest request path!");
        }

        XContentParser parser = request
            .getMediaType()
            .xContent()
            .createParser(request.getXContentRegistry(), LoggingDeprecationHandler.INSTANCE, content.streamInput());

        // XContentParser parser = request.contentParser();
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
        MLInput mlInput = MLInput.parse(parser, algorithm, actionType);
        if (FunctionName.REMOTE.name().contentEquals(algorithm)) {
            RemoteInferenceMLInput input = (RemoteInferenceMLInput) mlInput;
            RemoteInferenceInputDataSet inputDataSet = (RemoteInferenceInputDataSet) input.getInputDataset();
            inputDataSet.getParameters().put("stream", String.valueOf(true));
            return new MLPredictionTaskRequest(modelId, input, null, null);
        }
        return new MLPredictionTaskRequest(modelId, mlInput, null, null);
    }

    private HttpChunk createHttpChunkFromEvent(byte[] event) {
        BytesReference content = BytesReference.fromByteBuffer(ByteBuffer.wrap(event));
        return new HttpChunk() {
            @Override
            public void close() {
                if (content instanceof Releasable) {
                    ((Releasable) content).close();
                }
            }

            @Override
            public boolean isLast() {
                return false;
            }

            @Override
            public BytesReference content() {
                return content;
            }
        };
    }

    private HttpChunk convertToHttpChunk(MLTaskResponse response) throws IOException {
        String content = "";
        boolean isLast = false;

        // Extract content and is_last flag
        try {
            ModelTensorOutput output = (ModelTensorOutput) response.getOutput();
            if (output != null && !output.getMlModelOutputs().isEmpty()) {
                ModelTensors modelTensors = output.getMlModelOutputs().get(0);
                if (!modelTensors.getMlModelTensors().isEmpty()) {
                    Map<String, ?> dataMap = modelTensors.getMlModelTensors().get(0).getDataAsMap();
                    if (dataMap.containsKey("content")) {
                        content = (String) dataMap.get("content");
                        if (content == null) {
                            content = "";
                        }
                    }
                    if (dataMap.containsKey("is_last")) {
                        isLast = Boolean.TRUE.equals(dataMap.get("is_last"));
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to extract content from response", e);
            content = "";
        }

        log.info("Converting to HttpChunk - content: '{}', isLast: {}", content, isLast);

        // Create proper SSE formatted response
        String jsonData = "{\"content\":\"" + content.replace("\"", "\\\"") + "\",\"is_last\":" + isLast + "}";
        String sseData = "data: " + jsonData + "\n\n";
        BytesReference bytesRef = BytesReference.fromByteBuffer(ByteBuffer.wrap(sseData.getBytes()));

        return new HttpChunk() {
            @Override
            public void close() {
                if (bytesRef instanceof Releasable) {
                    ((Releasable) bytesRef).close();
                }
            }

            @Override
            public boolean isLast() {
                return false;
            }

            @Override
            public BytesReference content() {
                return bytesRef;
            }
        };
    }
}
