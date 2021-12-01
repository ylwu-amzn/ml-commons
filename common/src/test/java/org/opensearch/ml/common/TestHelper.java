/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 *
 */

package org.opensearch.ml.common;

import org.opensearch.common.Strings;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.ToXContentObject;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.common.xcontent.XContentType;

import java.io.IOException;
import java.util.function.Function;

public class TestHelper {

    public static <T> void testParse(ToXContentObject obj, Function<XContentParser, T> function) throws IOException {
        testParse(obj, function, false);
    }

    public static <T> void testParse(ToXContentObject obj, Function<XContentParser, T> function, boolean wrapWithObject) throws IOException {
        XContentBuilder builder = XContentFactory.contentBuilder(XContentType.JSON);
        if (wrapWithObject) {
            builder.startObject();
        }
        obj.toXContent(builder, ToXContent.EMPTY_PARAMS);
        if (wrapWithObject) {
            builder.endObject();
        }
        String jsonStr = Strings.toString(builder);
        testParseFromString(obj, jsonStr, function);
    }

    public static <T> void testParseFromString(ToXContentObject obj, String jsonStr, Function<XContentParser, T> function) throws IOException {
        XContentParser parser = XContentType.JSON.xContent().createParser(NamedXContentRegistry.EMPTY, LoggingDeprecationHandler.INSTANCE, jsonStr);
        parser.nextToken();
        T parsedObj = function.apply(parser);
        obj.equals(parsedObj);
    }
}
