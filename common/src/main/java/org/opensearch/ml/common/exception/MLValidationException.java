/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.exception;

import org.opensearch.common.io.stream.StreamInput;

import java.io.IOException;

/**
 * This exception is thrown when validation failed.
 */
public class MLValidationException extends MLException {

    public MLValidationException(StreamInput in) throws IOException {
        super(in);
    }
    /**
     * Constructor with error message.
     * @param message message of the exception
     */
    public MLValidationException(String message) {
        super(message);
    }

    /**
     * Constructor with specified cause.
     * @param cause exception cause
     */
    public MLValidationException(Throwable cause) {
        super(cause);
    }
}
