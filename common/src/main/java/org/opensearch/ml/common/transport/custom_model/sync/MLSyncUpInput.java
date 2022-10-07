/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom_model.sync;

import lombok.Builder;
import lombok.Data;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;

import java.io.IOException;
import java.util.Map;
import java.util.Set;

@Data
public class MLSyncUpInput implements Writeable {
    private static final String MODEL_ID_FIELD = "model_id";
    private static final String WORKER_NODES_FIELD = "worker_nodes";
    private String modelId;
    private String[] addedWorkerNodes;
    private String[] removedWorkerNodes;
    private boolean getLoadedModels;
    private Map<String, Set<String>> modelRoutingTable;

    public MLSyncUpInput(StreamInput in) throws IOException {
        this.modelId = in.readOptionalString();
        this.addedWorkerNodes = in.readOptionalStringArray();
        this.removedWorkerNodes = in.readOptionalStringArray();
        this.getLoadedModels = in.readBoolean();
        if (in.readBoolean()) {
            modelRoutingTable = in.readMap(StreamInput::readString, s -> s.readSet(StreamInput::readString));
        }
    }

    @Builder
    public MLSyncUpInput(String modelId, String[] addedWorkerNodes, String[] removedWorkerNodes, boolean getLoadedModels, Map<String, Set<String>> modelRoutingTable) {
        this.modelId = modelId;
        this.addedWorkerNodes = addedWorkerNodes;
        this.removedWorkerNodes = removedWorkerNodes;
        this.getLoadedModels = getLoadedModels;
        this.modelRoutingTable = modelRoutingTable;
    }

    public MLSyncUpInput() {}

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeOptionalString(modelId);
        out.writeOptionalStringArray(addedWorkerNodes);
        out.writeOptionalStringArray(removedWorkerNodes);
        out.writeBoolean(getLoadedModels);
        if (modelRoutingTable != null && modelRoutingTable.size() > 0) {
            out.writeBoolean(true);
            out.writeMap(modelRoutingTable, StreamOutput::writeString, StreamOutput::writeStringCollection);
        } else {
            out.writeBoolean(false);
        }
    }

}
