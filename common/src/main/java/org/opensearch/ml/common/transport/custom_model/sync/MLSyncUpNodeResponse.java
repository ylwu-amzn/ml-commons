/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom_model.sync;

import lombok.extern.log4j.Log4j2;
import org.opensearch.action.support.nodes.BaseNodeResponse;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;

import java.io.IOException;

@Log4j2
public class MLSyncUpNodeResponse extends BaseNodeResponse  {

    private String modelStatus;

    public MLSyncUpNodeResponse(DiscoveryNode node, String modelStatus) {
        super(node);
        this.modelStatus = modelStatus;
    }

    public MLSyncUpNodeResponse(StreamInput in) throws IOException {
        super(in);
        this.modelStatus = in.readOptionalString();
    }

    public static MLSyncUpNodeResponse readStats(StreamInput in) throws IOException {
        return new MLSyncUpNodeResponse(in);
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeOptionalString(modelStatus);
    }

}
