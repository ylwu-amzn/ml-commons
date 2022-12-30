package org.opensearch.ml.utils;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang3.exception.ExceptionUtils;
import org.apache.logging.log4j.Logger;
import org.opensearch.common.Strings;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.ml.common.exception.MLLimitExceededException;
import org.opensearch.ml.common.exception.MLResourceNotFoundException;

import java.io.IOException;
import java.util.Map;

//@Log4j2
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

    public static void logException(String errorMessage, Exception e, Logger log) {
        Throwable rootCause = ExceptionUtils.getRootCause(e);
        if (e instanceof MLLimitExceededException || e instanceof MLResourceNotFoundException) {
            log.warn(e.getMessage());
        } else if (rootCause instanceof MLLimitExceededException || rootCause instanceof MLResourceNotFoundException) {
            log.warn(rootCause.getMessage());
        } else {
            log.error(errorMessage, e);
        }
    }
}
