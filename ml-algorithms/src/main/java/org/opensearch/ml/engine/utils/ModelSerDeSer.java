/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.utils;

import io.protostuff.LinkedBuffer;
import io.protostuff.ProtostuffIOUtil;
import io.protostuff.Schema;
import lombok.experimental.UtilityClass;

@UtilityClass
public class ModelSerDeSer {
    public static final int SERIALIZATION_BUFFER_BYTES = 512;

    public static <T> byte[] serialize(T model, Schema<T> schema) {
        LinkedBuffer buffer = LinkedBuffer.allocate(SERIALIZATION_BUFFER_BYTES);
        return ProtostuffIOUtil.toByteArray(model, schema, buffer);
    }

    public static <T> T deserialize(byte[] bytes, Schema<T> schema) {
        T model = schema.newMessage();
        ProtostuffIOUtil.mergeFrom(bytes, model, schema);
        return model;
    }
}
