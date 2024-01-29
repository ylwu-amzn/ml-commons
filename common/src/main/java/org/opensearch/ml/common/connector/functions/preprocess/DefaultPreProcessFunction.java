package org.opensearch.ml.common.connector.functions.preprocess;

import lombok.AccessLevel;
import lombok.Builder;
import lombok.experimental.FieldDefaults;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.script.Script;
import org.opensearch.script.ScriptService;
import org.opensearch.script.ScriptType;
import org.opensearch.script.TemplateScript;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;

import static org.opensearch.core.xcontent.ToXContent.EMPTY_PARAMS;
import static org.opensearch.ml.common.utils.StringUtils.convertScriptStringToJsonString;
import static org.opensearch.ml.common.utils.StringUtils.gson;

@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class DefaultPreProcessFunction extends ConnectorPreProcessFunction {

    ScriptService scriptService;
    String preProcessFunction;

    @Builder
    public DefaultPreProcessFunction(ScriptService scriptService, String preProcessFunction) {
        this.returnDirectlyForRemoteInferenceInput = false;
        this.scriptService = scriptService;
        this.preProcessFunction = preProcessFunction;
    }

    @Override
    public void validate(MLInput mlInput) {
    }

    @Override
    public RemoteInferenceInputDataSet process(MLInput mlInput) {
        try (XContentBuilder builder = XContentFactory.jsonBuilder()) {
            mlInput.toXContent(builder, EMPTY_PARAMS);
            String inputStr = builder.toString();
            Map inputParams = Map.of("parameters", gson.fromJson(inputStr, Map.class));
            String processedInput = executeScript(scriptService, preProcessFunction, convertScriptStringToJsonString(inputParams));
            if (processedInput == null) {
                throw new IllegalArgumentException("Wrong input");
            }
            Map<String, Object> map = gson.fromJson(processedInput, Map.class);
            return RemoteInferenceInputDataSet.builder().parameters(convertScriptStringToJsonString(map)).build();
        } catch (IOException e) {
            throw new IllegalArgumentException("wrong ML input");
        }
    }

    private String executeScript(ScriptService scriptService, String painlessScript, Map<String, Object> params) {
        Script script = new Script(ScriptType.INLINE, "painless", painlessScript, Collections.emptyMap());
        TemplateScript templateScript = scriptService.compile(script, TemplateScript.CONTEXT).newInstance(params);
        return templateScript.execute();
    }
}
