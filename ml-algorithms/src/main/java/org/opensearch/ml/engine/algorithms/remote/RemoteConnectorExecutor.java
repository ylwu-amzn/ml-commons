/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.remote;

import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.script.ScriptService;

import java.util.List;

public interface RemoteConnectorExecutor {

    ModelTensorOutput execute(MLInput mlInput);

    default void setScriptService(ScriptService scriptService){}
    default void setClient(Client client){}
    default void setXContentRegistry(NamedXContentRegistry xContentRegistry){}
    default void setClusterService(ClusterService clusterService){}

    default void setTools(List<Tool> tools) {}

}
