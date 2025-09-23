package org.opensearch.ml.common.model.provider;

import com.fasterxml.jackson.core.JsonParseException;
import lombok.extern.log4j.Log4j2;
import org.opensearch.ml.common.MLCommonsClassLoader;
import org.opensearch.ml.common.connector.Connector;

import java.util.Map;

public interface ModelProvider {

    static ModelProvider createModelProvider(String providerName, String modelId, Map<String, String> credential, Map<String, Object> parameters) throws JsonParseException {
        return MLCommonsClassLoader.initModelProvider(providerName, new Object[] {modelId, credential, parameters}, String.class, Map.class, Map.class);
    }

    Connector createConnector();
}
