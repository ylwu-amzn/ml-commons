/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.remote;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import lombok.extern.log4j.Log4j2;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.connector.Connector;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.model.MLModelConfig;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.engine.MLEngineClassLoader;
import org.opensearch.ml.engine.algorithms.DLModel;
import org.opensearch.ml.engine.annotation.Function;
import org.opensearch.ml.engine.encryptor.Encryptor;
import org.opensearch.ml.engine.tools.SearchIndexTool;
import org.opensearch.script.ScriptService;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Log4j2
@Function(FunctionName.REMOTE)
public class RemoteModel extends DLModel {

    public static final String CLUSTER_SERVICE = "cluster_service";
    public static final String SCRIPT_SERVICE = "script_service";
    public static final String CLIENT = "client";
    public static final String XCONTENT_REGISTRY = "xcontent_registry";
    public static final String EXTERNAL_TOOLS = "external_tools";
    public static final String SETTINGS = "settings";
    private RemoteConnectorExecutor connectorExecutor;
    private List<Tool> tools;

    @Override
    public MLOutput predict(MLInput mlInput) {
        try {
            return predict(modelId, mlInput);
        } catch (Throwable t) {
            log.error("Failed to call remote model", t);
            throw new MLException("Failed to call remote model. " + t.getMessage());
        }
    }

    @Override
    public ModelTensorOutput predict(String modelId, MLInput mlInput) throws TranslateException {
        return connectorExecutor.execute(mlInput);
    }

    @Override
    public void initModel(MLModel model, Map<String, Object> params, Encryptor encryptor) {
        try {
            Connector connector = model.getConnector().clone();
            connector.decrypt((credential) -> encryptor.decrypt(credential));
            this.connectorExecutor = MLEngineClassLoader.initInstance(connector.getName(), connector, Connector.class);
            this.connectorExecutor.setScriptService((ScriptService) params.get(SCRIPT_SERVICE));
            this.connectorExecutor.setClusterService((ClusterService) params.get(CLUSTER_SERVICE));
            this.connectorExecutor.setClient((Client) params.get(CLIENT));
            this.connectorExecutor.setXContentRegistry((NamedXContentRegistry) params.get(XCONTENT_REGISTRY));

            Map<String, Tool> externalTools = (Map<String, Tool>)params.get(EXTERNAL_TOOLS);
            List<String> toolList = model.getTools();
            if (toolList != null && toolList.size() > 0) {
                tools = new ArrayList<>();
                for (String toolName : toolList) {
                    Tool tool;
                    if (externalTools.containsKey(toolName)) {
                        tool = externalTools.get(toolName);
                    } else {
                        tool = MLEngineClassLoader.initToolInstance(toolName, (ScriptService) params.get(SCRIPT_SERVICE), ScriptService.class);
                        if (tool instanceof SearchIndexTool) {
                            ((SearchIndexTool)tool).setClient((Client) params.get(CLIENT));
                            ((SearchIndexTool)tool).setXContentRegistry((NamedXContentRegistry) params.get(XCONTENT_REGISTRY));
                        }
                    }
                    tools.add(tool);
                }
            }
            this.connectorExecutor.setTools(tools);
        } catch (Exception e) {
            log.error("Failed to init remote model", e);
            throw new MLException(e);
        }
    }

    @Override
    public Translator<Input, Output> getTranslator(String engine, MLModelConfig modelConfig) {
        return null;
    }

    @Override
    public TranslatorFactory getTranslatorFactory(String engine, MLModelConfig modelConfig) {
        return null;
    }


}
