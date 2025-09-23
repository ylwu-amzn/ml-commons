package org.opensearch.ml.common.model.provider;

import lombok.Builder;
import lombok.NoArgsConstructor;
import org.opensearch.ml.common.annotation.ModelProvider;
import org.opensearch.ml.common.connector.AwsConnector;
import org.opensearch.ml.common.connector.Connector;
import org.opensearch.ml.common.connector.ConnectorAction;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@NoArgsConstructor
@ModelProvider("bedrock/converse")
public class BedrockConverse implements org.opensearch.ml.common.model.provider.ModelProvider {

    private String modelId;
    private Map<String, String> credential;
    private Map<String, String> parameters;

    @Builder
    public BedrockConverse(String modelId, Map<String, String> credential, Map<String, String> parameters) {
        this.modelId = modelId;
        if (credential == null) {
            throw new IllegalArgumentException("Credential not set");
        }
        this.credential = new HashMap<>();
        this.credential.putAll(credential);
        this.parameters = new HashMap<>();
        this.parameters.put("region", "us-east-1");
        this.parameters.put("service_name", "bedrock");

        if (parameters != null) {
            this.parameters.putAll(parameters);
        }
    }

    @Override
    public Connector createConnector() {
        List<ConnectorAction> actions = new ArrayList<>();
        String requestBody = """
        { 
          "system": [
            {"text": "${parameters.system_prompt}"}
          ],
          
          "messages": [
            ${parameters._chat_history:-}
            {
              "role":"user",
              "content": ${parameters.user_content_blocks}
            }
            ${parameters._interactions:-}
          ]
          
          ${parameters.tool_configs:-}
        }""";
        actions.add(ConnectorAction.builder()
                .actionType(ConnectorAction.ActionType.PREDICT)
                .method("POST")
                .url("https://bedrock-runtime."+parameters.get("region")+".amazonaws.com/model/"+modelId+"/converse")
                .headers(Map.of("content-type", "application/json"))
                .requestBody(requestBody)
                .build());
        return AwsConnector.awsConnectorBuilder().name(modelId).protocol("aws_sigv4").version("1.0").parameters(parameters).credential(credential).actions(actions).build();
    }
}
