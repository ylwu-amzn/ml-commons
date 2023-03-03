/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.utils;

import lombok.experimental.UtilityClass;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.serialization.ValidatingObjectInputStream;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.engine.exceptions.ModelSerDeSerException;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Base64;

@Log4j2
@UtilityClass
public class ModelSerDeSer {
    // Welcome list includes OpenSearch ml plugin classes, JDK common classes and Tribuo libraries.
    public static final String[] ACCEPT_CLASS_PATTERNS = {
            "java.lang.*",
            "java.util.*",
            "java.time.*",
            "org.opensearch.ml.*",
            "*org.tribuo.*",
            "libsvm.*",
            "com.oracle.labs.*",
            "[*",
            "com.amazon.randomcutforest.*"
    };

    public static String serializeToBase64(Object model) {
        byte[] bytes = serialize(model);
        return encodeBase64(bytes);
    }

    public static byte[] serialize(Object model) {
        try (ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
             ObjectOutputStream objectOutputStream = new ObjectOutputStream(byteArrayOutputStream)) {
            objectOutputStream.writeObject(model);
            objectOutputStream.flush();
            return byteArrayOutputStream.toByteArray();
        } catch (IOException e) {
            throw new ModelSerDeSerException("Failed to serialize model.", e.getCause());
        }
    }

    public static Object deserialize(byte[] modelBin) {
        ClassLoader contextClassLoader = Thread.currentThread().getContextClassLoader();
//        Thread.currentThread().setContextClassLoader(ValidatingObjectInputStream.class.getClassLoader());
        try (ByteArrayInputStream inputStream = new ByteArrayInputStream(modelBin);
             ObjectInputStream validatingObjectInputStream = new ObjectInputStream(inputStream)) {
//             ValidatingObjectInputStream validatingObjectInputStream = new ValidatingObjectInputStream(inputStream)) {
            // Validate the model class type to avoid deserialization attack.
//            validatingObjectInputStream.accept(ACCEPT_CLASS_PATTERNS);
            return validatingObjectInputStream.readObject();
        } catch (Throwable e) {
            log.error("-----1 Failed to deserializea model ", e);
            throw new ModelSerDeSerException("Failed to deserialize model.", e.getCause());
        } finally {
            Thread.currentThread().setContextClassLoader(contextClassLoader);
        }
    }

    public static Object deserialize(MLModel model) {
        byte[] decodeBytes = decodeBase64(model.getContent());
        return deserialize(decodeBytes);
    }

    public static byte[] decodeBase64(String base64Str) {
        return Base64.getDecoder().decode(base64Str);
    }

    public static String encodeBase64(byte[] bytes) {
        return Base64.getEncoder().encodeToString(bytes);
    }
}
