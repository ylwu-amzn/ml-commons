/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.spi.memory;

/**
 * A general memory interface.
 * @param <T>
 */
public interface Memory<T extends Message> {

    /**
     * Get memory type.
     * @return
     */
    String getType();

    /**
     * Save message to id.
     * @param id memory id
     * @param message message to be saved
     */
    void save(String id, T message);

    /**
     * Get messages of memory id.
     * @param id memory id
     * @return
     */
    T[] getMessages(String id);

    /**
     * Clear all memory.
     */
    void clear();

    /**
     * Remove memory of specific id.
     * @param id memory id
     */
    void remove(String id);
}
