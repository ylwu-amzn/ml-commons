/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common;

import org.opensearch.common.Strings;
import org.opensearch.common.bytes.BytesReference;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.ToXContentObject;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.search.SearchModule;

import java.io.IOException;
import java.util.Collections;
import java.util.function.Function;

public class TestHelper {

    public static XContentParser parser(String xc) throws IOException {
        return parser(xc, true);
    }

    public static XContentParser parser(String xc, boolean skipFirstToken) throws IOException {
        XContentParser parser = XContentType.JSON.xContent().createParser(xContentRegistry(), LoggingDeprecationHandler.INSTANCE, xc);
        if (skipFirstToken) {
            parser.nextToken();
        }
        return parser;
    }

    public static NamedXContentRegistry xContentRegistry() {
        SearchModule searchModule = new SearchModule(Settings.EMPTY, Collections.emptyList());
        return new NamedXContentRegistry(searchModule.getNamedXContents());
    }

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

    public static String contentObjectToString(ToXContentObject obj) throws IOException {
        XContentBuilder builder = XContentFactory.contentBuilder(XContentType.JSON);
        obj.toXContent(builder, ToXContent.EMPTY_PARAMS);
        return xContentBuilderToString(builder);
    }

    public static String xContentBuilderToString(XContentBuilder builder) {
        return BytesReference.bytes(builder).utf8ToString();
    }
}
