/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.remote;

import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.connector.AwsConnector;
import org.opensearch.ml.common.connector.Connector;
import org.opensearch.ml.common.dataset.TextDocsInputDataSet;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.engine.annotation.ConnectorExecutor;
import org.opensearch.script.ScriptService;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.signer.Aws4Signer;
import software.amazon.awssdk.auth.signer.params.Aws4SignerParams;
import software.amazon.awssdk.core.internal.http.loader.DefaultSdkHttpClientBuilder;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.http.AbortableInputStream;
import software.amazon.awssdk.http.HttpExecuteRequest;
import software.amazon.awssdk.http.HttpExecuteResponse;
import software.amazon.awssdk.http.SdkHttpClient;
import software.amazon.awssdk.http.SdkHttpFullRequest;
import software.amazon.awssdk.regions.Region;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.AccessController;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.ml.common.connector.ConnectorNames.AWS_V1;
import static org.opensearch.ml.engine.algorithms.remote.ConnectorUtils.processInput;
import static org.opensearch.ml.engine.algorithms.remote.ConnectorUtils.processOutput;
import static software.amazon.awssdk.http.SdkHttpMethod.POST;

@Log4j2
@ConnectorExecutor(AWS_V1)
public class AwsConnectorExecutor implements RemoteConnectorExecutor{

    private AwsConnector connector;
    private final Aws4Signer signer;
    private final SdkHttpClient httpClient;
    @Setter
    private ScriptService scriptService;

    public AwsConnectorExecutor(Connector connector) {
        this.connector = (AwsConnector)connector;
        this.signer = Aws4Signer.create();
        this.httpClient = new DefaultSdkHttpClientBuilder().build();
    }

    @Override
    public ModelTensorOutput execute(MLInput mlInput) {
        List<ModelTensors> tensorOutputs = new ArrayList<>();

        if (mlInput.getInputDataset() instanceof TextDocsInputDataSet) {
            TextDocsInputDataSet textDocsInputDataSet = (TextDocsInputDataSet) mlInput.getInputDataset();
            List textDocs = new ArrayList(textDocsInputDataSet.getDocs());
            for (int i = 0; i < textDocsInputDataSet.getDocs().size(); i++) {
                List<ModelTensor> modelTensors = new ArrayList<>();
                getModelTensorOutput(MLInput.builder().algorithm(FunctionName.TEXT_EMBEDDING).inputDataset(TextDocsInputDataSet.builder().docs(textDocs).build()).build(), tensorOutputs, modelTensors);
                if (tensorOutputs.size() >= textDocsInputDataSet.getDocs().size()) {
                    break;
                }
                textDocs.remove(0);
            }
        } else {
            List<ModelTensor> modelTensors = new ArrayList<>();
            getModelTensorOutput(mlInput, tensorOutputs, modelTensors);
        }
        return new ModelTensorOutput(tensorOutputs);
    }

    private void getModelTensorOutput(MLInput mlInput, List<ModelTensors> tensorOutputs, List<ModelTensor> modelTensors) {
        try {
            RemoteInferenceInputDataSet inputData = processInput(mlInput, connector, scriptService);

            Map<String, String> parameters = new HashMap<>();
            if (connector.getParameters() != null) {
                parameters.putAll(connector.getParameters());
            }
            if (inputData.getParameters() != null) {
                parameters.putAll(inputData.getParameters());
            }

            String payload = connector.createPayload(parameters);
            connector.validatePayload(payload);

            String endpoint = connector.getEndpoint();
            RequestBody requestBody = RequestBody.fromString(payload);

            SdkHttpFullRequest.Builder builder = SdkHttpFullRequest.builder()
                    .method(POST)
                    .uri(URI.create(endpoint))
                    .contentStreamProvider(requestBody.contentStreamProvider());
            Map<String, String> headers = connector.createHeaders();
            for (String key : headers.keySet()) {
                builder.putHeader(key, headers.get(key));
            }
            SdkHttpFullRequest request = builder.build();
            HttpExecuteRequest executeRequest = HttpExecuteRequest.builder()
                    .request(signRequest(request))
                    .contentStreamProvider(request.contentStreamProvider().orElse(null))
                    .build();

            HttpExecuteResponse response = AccessController.doPrivileged((PrivilegedExceptionAction<HttpExecuteResponse>) () -> {
                return httpClient.prepareRequest(executeRequest).call();
            });

            AbortableInputStream body = null;
            if (response.responseBody().isPresent()) {
                body = response.responseBody().get();
            }

            StringBuilder responseBuilder = new StringBuilder();
            if (body != null) {
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(body, StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        responseBuilder.append(line);
                    }
                }
            }
            String modelResponse = responseBuilder.toString();

            ModelTensors tensors = processOutput(modelResponse, connector, scriptService, parameters, modelTensors);
            tensorOutputs.add(tensors);
        } catch (Throwable e) {
            log.error("Failed to execute aws connector", e);
            throw new MLException("Fail to execute aws connector", e);
        }
    }

    private SdkHttpFullRequest signRequest(SdkHttpFullRequest request) {
        String accessKey = connector.getAccessKey();
        String secretKey = connector.getSecretKey();
        String signingName = connector.getServiceName();
        String region = connector.getRegion();
        AwsBasicCredentials credentials = AwsBasicCredentials.create(accessKey, secretKey);

        Aws4SignerParams params = Aws4SignerParams.builder()
                .awsCredentials(credentials)
                .signingName(signingName)
                .signingRegion(Region.of(region))
                .build();

        return signer.sign(request, params);
    }
}
