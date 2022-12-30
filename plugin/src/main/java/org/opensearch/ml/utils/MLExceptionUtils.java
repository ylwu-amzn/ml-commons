package org.opensearch.ml.utils;

import org.apache.commons.lang3.exception.ExceptionUtils;
import org.opensearch.common.Strings;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;

import java.io.IOException;
import java.util.Map;

public class MLExceptionUtils {

    public static final String NOT_SERIALIZABLE_EXCEPTION_WRAPPER = "NotSerializableExceptionWrapper: ";

    public static String getRootCauseMessage(final Throwable throwable) {
        String message = ExceptionUtils.getRootCauseMessage(throwable);
        if (message != null && message.startsWith(NOT_SERIALIZABLE_EXCEPTION_WRAPPER)) {
            message = message.replace(NOT_SERIALIZABLE_EXCEPTION_WRAPPER, "");
        }
        message = message.substring(message.indexOf(":")+2);
        return message;
    }

    public static String toJsonString(Map<String, String> nodeErrors) {
        if (nodeErrors == null || nodeErrors.size() == 0) {
            return null;
        }
        try {
            XContentBuilder builder = XContentFactory.jsonBuilder();
            builder.startObject();
            for (Map.Entry<String, String> entry : nodeErrors.entrySet()) {
                builder.field(entry.getKey(), entry.getValue());
            }
            builder.endObject();
            return Strings.toString(builder);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
