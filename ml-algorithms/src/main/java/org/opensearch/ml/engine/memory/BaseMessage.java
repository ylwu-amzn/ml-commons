/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.memory;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.opensearch.ml.common.spi.memory.Message;

public class BaseMessage implements Message {

    @Getter
    @Setter
    protected String type;
    @Getter
    @Setter
    protected String content;

    @Builder
    public BaseMessage(String type, String content) {
        this.type = type;
        this.content = content;
    }

    @Override
    public String toString() {
        return type + ": " + content;
    }
}
