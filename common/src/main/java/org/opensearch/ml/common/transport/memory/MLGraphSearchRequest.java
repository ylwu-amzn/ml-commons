/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.memory;

import static org.opensearch.action.ValidateActions.addValidationError;
import static org.opensearch.core.xcontent.XContentParserUtils.ensureExpectedToken;

import java.io.IOException;

import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.xcontent.XContentParser;

import lombok.Builder;
import lombok.Getter;

/**
 * Request to search graph data within a memory container
 */
@Getter
public class MLGraphSearchRequest extends ActionRequest {

    private MLGraphSearchInput input;
    private String tenantId;

    @Builder
    public MLGraphSearchRequest(MLGraphSearchInput input, String tenantId) {
        this.input = input;
        this.tenantId = tenantId;
    }

    public MLGraphSearchRequest(StreamInput in) throws IOException {
        super(in);
        this.input = new MLGraphSearchInput(in);
        this.tenantId = in.readOptionalString();
    }

    public static MLGraphSearchRequest parse(XContentParser parser, String tenantId) throws IOException {
        // Advance to the first token if the parser is at its initial (null) position.
        if (parser.currentToken() == null) {
            parser.nextToken();
        }
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        MLGraphSearchInput input = MLGraphSearchInput.parse(parser);
        return MLGraphSearchRequest.builder()
            .input(input)
            .tenantId(tenantId)
            .build();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        input.writeTo(out);
        out.writeOptionalString(tenantId);
    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException exception = null;

        if (input == null) {
            exception = addValidationError("Graph search input is required", exception);
        } else {
            if (input.getMemoryContainerId() == null) {
                exception = addValidationError("Memory container ID is required", exception);
            }
            if (input.getQuery() == null || input.getQuery().trim().isEmpty()) {
                exception = addValidationError("Search query is required", exception);
            }
        }

        return exception;
    }
}