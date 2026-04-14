/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.memory;

import static org.opensearch.action.ValidateActions.addValidationError;

import java.io.IOException;

import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;

import lombok.Builder;
import lombok.Getter;

/**
 * Request to delete graph data from a memory container
 */
@Getter
public class MLDeleteGraphDataRequest extends ActionRequest {

    private String memoryContainerId;
    private String tenantId;

    @Builder
    public MLDeleteGraphDataRequest(String memoryContainerId, String tenantId) {
        this.memoryContainerId = memoryContainerId;
        this.tenantId = tenantId;
    }

    public MLDeleteGraphDataRequest(StreamInput in) throws IOException {
        super(in);
        this.memoryContainerId = in.readString();
        this.tenantId = in.readOptionalString();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(memoryContainerId);
        out.writeOptionalString(tenantId);
    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException exception = null;

        if (memoryContainerId == null || memoryContainerId.trim().isEmpty()) {
            exception = addValidationError("Memory container ID is required", exception);
        }

        return exception;
    }
}