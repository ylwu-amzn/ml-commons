/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom_model.unload;

import lombok.Getter;
import org.opensearch.action.support.nodes.BaseNodesRequest;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;

import java.io.IOException;

public class UnloadModelNodesRequest extends BaseNodesRequest<UnloadModelNodesRequest> {

    @Getter
    private UnloadModelInput unloadModelInput;

    public UnloadModelNodesRequest(StreamInput in) throws IOException {
        super(in);
        unloadModelInput = new UnloadModelInput(in);
    }

    public UnloadModelNodesRequest(String[] nodeIds, UnloadModelInput unloadModelInput) {
        super(nodeIds);
        this.unloadModelInput = unloadModelInput;
    }

    public UnloadModelNodesRequest(DiscoveryNode... nodes) {
        super(nodes);
        unloadModelInput = new UnloadModelInput();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        unloadModelInput.writeTo(out);
    }

}
