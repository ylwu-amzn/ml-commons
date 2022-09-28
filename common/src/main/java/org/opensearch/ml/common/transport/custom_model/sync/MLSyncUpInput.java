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

@Data
public class MLSyncUpInput implements Writeable {
    private static final String MODEL_ID_FIELD = "model_id";
    private static final String WORKER_NODES_FIELD = "worker_nodes";
    private String modelId;
    private String[] addedWorkerNodes;
    private String[] removedWorkerNOdes;

    public MLSyncUpInput(StreamInput in) throws IOException {
        this.modelId = in.readString();
        this.addedWorkerNodes = in.readOptionalStringArray();
        this.removedWorkerNOdes = in.readOptionalStringArray();
    }

    @Builder
    public MLSyncUpInput(String modelId, String[] addedWorkerNodes, String[] removedWorkerNOdes) {
        this.modelId = modelId;
        this.addedWorkerNodes = addedWorkerNodes;
        this.removedWorkerNOdes = removedWorkerNOdes;
    }

    public MLSyncUpInput() {}

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(modelId);
        out.writeOptionalStringArray(addedWorkerNodes);
        out.writeOptionalStringArray(removedWorkerNOdes);
    }

}
