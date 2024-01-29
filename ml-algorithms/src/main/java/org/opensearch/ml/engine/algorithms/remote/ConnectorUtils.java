/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.remote;

import static org.apache.commons.text.StringEscapeUtils.escapeJson;
import static org.opensearch.core.xcontent.ToXContent.EMPTY_PARAMS;
import static org.opensearch.ml.common.connector.HttpConnector.RESPONSE_FILTER_FIELD;
import static org.opensearch.ml.common.utils.StringUtils.gson;
import static org.opensearch.ml.engine.utils.ScriptUtils.executeBuildInPostProcessFunction;
import static org.opensearch.ml.engine.utils.ScriptUtils.executePostProcessFunction;
import static org.opensearch.ml.engine.utils.ScriptUtils.executeScript;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.text.StringSubstitutor;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.ml.common.connector.Connector;
import org.opensearch.ml.common.connector.ConnectorAction;
import org.opensearch.ml.common.connector.MLPostProcessFunction;
import org.opensearch.ml.common.connector.MLPreProcessFunction;
import org.opensearch.ml.common.connector.functions.preprocess.DefaultPreProcessFunction;
import org.opensearch.ml.common.connector.functions.preprocess.RemoteInferencePreProcessFunction;
import org.opensearch.ml.common.dataset.TextDocsInputDataSet;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.script.ScriptService;

import com.jayway.jsonpath.JsonPath;

import lombok.extern.log4j.Log4j2;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.AwsCredentials;
import software.amazon.awssdk.auth.credentials.AwsSessionCredentials;
import software.amazon.awssdk.auth.signer.Aws4Signer;
import software.amazon.awssdk.auth.signer.params.Aws4SignerParams;
import software.amazon.awssdk.http.SdkHttpFullRequest;
import software.amazon.awssdk.regions.Region;
import static org.opensearch.ml.common.utils.StringUtils.convertScriptStringToJsonString;

@Log4j2
public class ConnectorUtils {

    private static final Aws4Signer signer;
    static {
        signer = Aws4Signer.create();
    }

    public static RemoteInferenceInputDataSet processInput(
        MLInput mlInput,
        Connector connector,
        Map<String, String> parameters,
        ScriptService scriptService
    ) {
        if (mlInput == null) {
            throw new IllegalArgumentException("Input is null");
        }
        Optional<ConnectorAction> predictAction = connector.findPredictAction();
        if (predictAction.isEmpty()) {
            throw new IllegalArgumentException("no predict action found");
        }
        RemoteInferenceInputDataSet inputData = processMLInput(mlInput, connector, parameters, scriptService);
        if (inputData.getParameters() != null) {
            Map<String, String> newParameters = new HashMap<>();
            inputData.getParameters().forEach((key, value) -> {
                if (value == null) {
                    newParameters.put(key, null);
                } else if (org.opensearch.ml.common.utils.StringUtils.isJson(value)) {
                    // no need to escape if it's already valid json
                    newParameters.put(key, value);
                } else {
                    newParameters.put(key, escapeJson(value));
                }
            });
            inputData.setParameters(newParameters);
        }
        return inputData;
    }

    private static RemoteInferenceInputDataSet processMLInput(
        MLInput mlInput,
        Connector connector,
        Map<String, String> parameters,
        ScriptService scriptService
    ) {
        String preProcessFunction = getPreprocessFunction(mlInput, connector);
        if (preProcessFunction == null) {
            if (mlInput.getInputDataset() instanceof RemoteInferenceInputDataSet) {
                return (RemoteInferenceInputDataSet) mlInput.getInputDataset();
            } else {
                throw new IllegalArgumentException("pre_process_function not defined in connector");
            }
        } else {
            preProcessFunction = fillProcessFunctionParameter(parameters, preProcessFunction);
            if (MLPreProcessFunction.contains(preProcessFunction)) {
                Function<MLInput, RemoteInferenceInputDataSet> function = MLPreProcessFunction.get(preProcessFunction);
                return function.apply(mlInput);
            } else if (mlInput.getInputDataset() instanceof RemoteInferenceInputDataSet) {
                RemoteInferencePreProcessFunction function = new RemoteInferencePreProcessFunction(scriptService, preProcessFunction);
                return function.apply(mlInput);
            } else {
//                DefaultPreProcessFunction function = null;
//                try (XContentBuilder builder = XContentFactory.jsonBuilder()) {
//                    function = new DefaultPreProcessFunction(scriptService, preProcessFunction, builder);
//                } catch (IOException e) {
//                    throw new RuntimeException(e);
//                }
//                return function.apply(mlInput);

//                try (XContentBuilder builder = XContentBuilder.builder(XContentType.JSON.xContent());) {
//                try (XContentBuilder builder = JsonXContent.contentBuilder()) {
//                try (XContentBuilder builder = XContentBuilder.builder(JsonXContent.jsonXContent)) {
                try (XContentBuilder builder = MediaTypeRegistry.JSON.contentBuilder()) {
                    mlInput.toXContent(builder, EMPTY_PARAMS);
                    String inputStr = builder.toString();
                    Map inputParams = gson.fromJson(inputStr, Map.class);
                    String processedInput = executeScript(scriptService, preProcessFunction, inputParams);
                    if (processedInput == null) {
                        throw new IllegalArgumentException("Wrong input");
                    }
                    Map<String, Object> map = gson.fromJson(processedInput, Map.class);
                    return RemoteInferenceInputDataSet.builder().parameters(convertScriptStringToJsonString(map)).build();
                } catch (IOException e) {
                    throw new IllegalArgumentException("wrong ML input");
                }
            }
        }
    }

    // private static RemoteInferenceInputDataSet processTextDocsInput(
    // TextDocsInputDataSet inputDataSet,
    // Connector connector,
    // Map<String, String> parameters,
    // ScriptService scriptService
    // ) {
    // Optional<ConnectorAction> predictAction = connector.findPredictAction();
    // String preProcessFunction = predictAction.get().getPreProcessFunction();
    // preProcessFunction = preProcessFunction == null ? MLPreProcessFunction.TEXT_DOCS_TO_DEFAULT_EMBEDDING_INPUT : preProcessFunction;
    // if (MLPreProcessFunction.contains(preProcessFunction)) {
    // Function<?, Map<String, Object>> function = MLPreProcessFunction.get(preProcessFunction);
    // Map<String, Object> buildInFunctionResult = ((Function<List<String>, Map<String, Object>>) function)
    // .apply(inputDataSet.getDocs());
    // return RemoteInferenceInputDataSet.builder().parameters(convertScriptStringToJsonString(buildInFunctionResult)).build();
    // } else {
    // List<String> docs = new ArrayList<>();
    // for (String doc : inputDataSet.getDocs()) {
    // if (doc != null) {
    // String gsonString = gson.toJson(doc);
    // // in 2.9, user will add " before and after string
    // // gson.toString(string) will add extra " before after string, so need to remove
    // docs.add(gsonString.substring(1, gsonString.length() - 1));
    // } else {
    // docs.add(null);
    // }
    // }
    // if (preProcessFunction.contains("${parameters.")) {
    // StringSubstitutor substitutor = new StringSubstitutor(parameters, "${parameters.", "}");
    // preProcessFunction = substitutor.replace(preProcessFunction);
    // }
    // Optional<String> processedInput = executePreprocessFunction(scriptService, preProcessFunction, docs);
    // if (processedInput.isEmpty()) {
    // throw new IllegalArgumentException("Wrong input");
    // }
    // Map<String, Object> map = gson.fromJson(processedInput.get(), Map.class);
    // return RemoteInferenceInputDataSet.builder().parameters(convertScriptStringToJsonString(map)).build();
    // }
    // }

    private static String getPreprocessFunction(MLInput mlInput, Connector connector) {
        Optional<ConnectorAction> predictAction = connector.findPredictAction();
        String preProcessFunction = predictAction.get().getPreProcessFunction();
        if (preProcessFunction != null) {
            return preProcessFunction;
        }
        if (mlInput.getInputDataset() instanceof TextDocsInputDataSet) {
            return MLPreProcessFunction.TEXT_DOCS_TO_DEFAULT_EMBEDDING_INPUT;
        }
        return null;
    }

    // private static RemoteInferenceInputDataSet processTextSimilarityInput(
    // TextSimilarityInputDataSet inputDataSet,
    // Connector connector,
    // Map<String, String> parameters,
    // ScriptService scriptService
    // ) {
    // Optional<ConnectorAction> predictAction = connector.findPredictAction();
    // String preProcessFunction = predictAction.get().getPreProcessFunction();
    // preProcessFunction = preProcessFunction == null ? MLPreProcessFunction.TEXT_SIMILARITY_TO_DEFAULT_INPUT : preProcessFunction;
    // if (MLPreProcessFunction.contains(preProcessFunction)) {
    // Function<TextSimilarityInputDataSet, Map<String, Object>> function = MLPreProcessFunction.get(preProcessFunction);
    // Map<String, Object> buildInFunctionResult = function.apply(inputDataSet);
    // return RemoteInferenceInputDataSet.builder().parameters(convertScriptStringToJsonString(buildInFunctionResult)).build();
    // } else {
    // List<String> docs = new ArrayList<>();
    // for (String doc : inputDataSet.getTextDocs()) {
    // if (doc != null) {
    // String gsonString = gson.toJson(doc);
    // // in 2.9, user will add " before and after string
    // // gson.toString(string) will add extra " before after string, so need to remove
    // docs.add(gsonString.substring(1, gsonString.length() - 1));
    // } else {
    // docs.add(null);
    // }
    // }
    // String query = gson.toJson(inputDataSet.getQueryText());
    // query = query.substring(1, query.length() - 1);
    // if (preProcessFunction.contains("${parameters.")) {
    // StringSubstitutor substitutor = new StringSubstitutor(parameters, "${parameters.", "}");
    // preProcessFunction = substitutor.replace(preProcessFunction);
    // }
    // String processedInput = executeScript(scriptService, preProcessFunction, Map.of("query_text", query, "text_docs", docs));
    // if (processedInput == null) {
    // throw new IllegalArgumentException("Wrong input");
    // }
    // Map<String, Object> map = gson.fromJson(processedInput, Map.class);
    // return RemoteInferenceInputDataSet.builder().parameters(convertScriptStringToJsonString(map)).build();
    // }
    // }

    // private static Map<String, String> convertScriptStringToJsonString(Map<String, Object> processedInput) {
    // Map<String, String> parameterStringMap = new HashMap<>();
    // try {
    // AccessController.doPrivileged((PrivilegedExceptionAction<Void>) () -> {
    // Map<String, Object> parametersMap = (Map<String, Object>) processedInput.get("parameters");
    // for (String key : parametersMap.keySet()) {
    // if (parametersMap.get(key) instanceof String) {
    // parameterStringMap.put(key, (String) parametersMap.get(key));
    // } else {
    // parameterStringMap.put(key, gson.toJson(parametersMap.get(key)));
    // }
    // }
    // return null;
    // });
    // } catch (PrivilegedActionException e) {
    // log.error("Error processing parameters", e);
    // throw new RuntimeException(e);
    // }
    // return parameterStringMap;
    // }

    public static ModelTensors processOutput(
        String modelResponse,
        Connector connector,
        ScriptService scriptService,
        Map<String, String> parameters
    ) throws IOException {
        if (modelResponse == null) {
            throw new IllegalArgumentException("model response is null");
        }
        List<ModelTensor> modelTensors = new ArrayList<>();
        Optional<ConnectorAction> predictAction = connector.findPredictAction();
        if (predictAction.isEmpty()) {
            throw new IllegalArgumentException("no predict action found");
        }
        ConnectorAction connectorAction = predictAction.get();
        String postProcessFunction = connectorAction.getPostProcessFunction();
        postProcessFunction = fillProcessFunctionParameter(parameters, postProcessFunction);

        String responseFilter = parameters.get(RESPONSE_FILTER_FIELD);
        if (MLPostProcessFunction.contains(postProcessFunction)) {
            // in this case, we can use jsonpath to build a List<List<Float>> result from model response.
            if (StringUtils.isBlank(responseFilter))
                responseFilter = MLPostProcessFunction.getResponseFilter(postProcessFunction);

            Object filteredOutput = JsonPath.read(modelResponse, responseFilter);
            List<ModelTensor> processedResponse = executeBuildInPostProcessFunction(
                filteredOutput,
                MLPostProcessFunction.get(postProcessFunction)
            );
            return ModelTensors.builder().mlModelTensors(processedResponse).build();
        }

        // execute user defined painless script.
        Optional<String> processedResponse = executePostProcessFunction(scriptService, postProcessFunction, modelResponse);
        String response = processedResponse.orElse(modelResponse);
        boolean scriptReturnModelTensor = postProcessFunction != null
            && processedResponse.isPresent()
            && org.opensearch.ml.common.utils.StringUtils.isJson(response);
        if (responseFilter == null) {
            connector.parseResponse(response, modelTensors, scriptReturnModelTensor);
        } else {
            Object filteredResponse = JsonPath.parse(response).read(parameters.get(RESPONSE_FILTER_FIELD));
            connector.parseResponse(filteredResponse, modelTensors, scriptReturnModelTensor);
        }
        return ModelTensors.builder().mlModelTensors(modelTensors).build();
    }

    private static String fillProcessFunctionParameter(Map<String, String> parameters, String processFunction) {
        if (processFunction != null && processFunction.contains("${parameters.")) {
            Map<String, String> tmpParameters = new HashMap<>();
            for (String key : parameters.keySet()) {
                tmpParameters.put(key, gson.toJson(parameters.get(key)));
            }
            StringSubstitutor substitutor = new StringSubstitutor(tmpParameters, "${parameters.", "}");
            processFunction = substitutor.replace(processFunction);
        }
        return processFunction;
    }

    public static SdkHttpFullRequest signRequest(
        SdkHttpFullRequest request,
        String accessKey,
        String secretKey,
        String sessionToken,
        String signingName,
        String region
    ) {
        AwsCredentials credentials = sessionToken == null
            ? AwsBasicCredentials.create(accessKey, secretKey)
            : AwsSessionCredentials.create(accessKey, secretKey, sessionToken);

        Aws4SignerParams params = Aws4SignerParams
            .builder()
            .awsCredentials(credentials)
            .signingName(signingName)
            .signingRegion(Region.of(region))
            .build();

        return signer.sign(request, params);
    }
}
