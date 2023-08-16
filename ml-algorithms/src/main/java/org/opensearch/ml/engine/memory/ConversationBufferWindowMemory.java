/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.memory;

import lombok.Getter;
import lombok.Setter;
import org.opensearch.ml.common.spi.memory.Message;

import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;

/**
 * This memory is for storing conversation history in memory.
 * It stores latest N messages which defined in window size.
 */
public class ConversationBufferWindowMemory extends BufferMemory {
    public static final String TYPE = "conversation_buffer_widow";

    @Getter @Setter
    private int windowSize = 5;

    public ConversationBufferWindowMemory() {
        this.messages = new ConcurrentHashMap<>();
    }

    @Override
    public String getType() {
        return TYPE;
    }

    @Override
    public void save(String sessionId, Message message) {
        Queue<Message> messageQueue = this.messages.computeIfAbsent(sessionId, k -> new ConcurrentLinkedDeque<>());
        while (messageQueue.size() >= windowSize) {
            messageQueue.poll();
        }
        messageQueue.add(message);
    }
}
