/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.agent;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.text.StringSubstitutor;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.agent.MLAgent;
import org.opensearch.ml.common.agent.MLToolSpec;
import org.opensearch.ml.common.spi.tools.Tool;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Log4j2
@Data
@NoArgsConstructor
public class MLFlowAgentRunner {

    private Client client;
    private Settings settings;
    private ClusterService clusterService;
    private NamedXContentRegistry xContentRegistry;
    private Map<String, Tool.Factory> toolFactories;

    public MLFlowAgentRunner(Client client, Settings settings, ClusterService clusterService, NamedXContentRegistry xContentRegistry, Map<String, Tool.Factory> toolFactories) {
        this.client = client;
        this.settings = settings;
        this.clusterService = clusterService;
        this.xContentRegistry = xContentRegistry;
        this.toolFactories = toolFactories;
    }

    public Object run(MLAgent mlAgent, Map<String, String> params) {
        List<MLToolSpec> toolSpecs = mlAgent.getTools();
        Object lastStepOutput = null;
        for (int i = 0 ;i<toolSpecs.size(); i++) {
            MLToolSpec toolSpec = toolSpecs.get(i);
            Map<String, String> toolParams = new HashMap<>();
            Map<String, String> executeParams = new HashMap<>();
            if (toolSpec.getParameters() != null) {
                toolParams.putAll(toolSpec.getParameters());
                executeParams.putAll(toolSpec.getParameters());
            }
            for (String key : params.keySet()) {
                if (key.startsWith(toolSpec.getName() + ".")) {
                    executeParams.put(key.replace(toolSpec.getName()+".", ""), params.get(key));
                } else {
                    executeParams.put(key, params.get(key));
                }
            }
            Tool tool = toolFactories.get(toolSpec.getName()).create(toolParams);
            tool.setAlias(toolSpec.getAlias());

            if (toolSpec.getDescription() != null) {
                tool.setDescription(toolSpec.getDescription());
            }

            if (executeParams.containsKey("input")) {
                String input = executeParams.get("input");
                StringSubstitutor substitutor = new StringSubstitutor(executeParams, "${parameters.", "}");
                input = substitutor.replace(input);
                executeParams.put("input", input);
            }

            if (i == 0) {
                lastStepOutput  = tool.run(executeParams);
                params.put(toolSpec.getName() + ".output", lastStepOutput + "");
            } else {
                lastStepOutput  = tool.run(executeParams);
            }
        }
        return lastStepOutput;
    }

}
