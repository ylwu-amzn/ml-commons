/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools;

import com.google.common.collect.ImmutableMap;
import lombok.extern.log4j.Log4j2;
import org.opensearch.action.ActionListener;
import org.opensearch.action.LatchedActionListener;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.client.Client;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.spi.tools.ToolAnnotation;
import org.opensearch.search.SearchHit;
import org.opensearch.search.builder.SearchSourceBuilder;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedExceptionAction;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.opensearch.ml.engine.utils.ScriptUtils.gson;

@Log4j2
@ToolAnnotation(SearchIndexTool.NAME)
public class SearchIndexTool implements Tool {
    public static final String NAME = "SearchIndexTool";

    private static final String description = "Use this tool to query OpenSearch index.";

    private Client client;
    private NamedXContentRegistry xContentRegistry;

    @Override
    public <T> T run(String input) {
        Map<String, String> parameters = parseInput(input);
        String index = parameters.get("index");
        String query = parameters.get("query");

        try {
            AtomicReference<String> contextRef = new AtomicReference<>("");
            AtomicReference<Exception> exceptionRef = new AtomicReference<>(null);

            SearchSourceBuilder searchSourceBuilder = new SearchSourceBuilder();
            XContentParser queryParser = XContentType.JSON.xContent().createParser(xContentRegistry, LoggingDeprecationHandler.INSTANCE, query);
            searchSourceBuilder.parseXContent(queryParser);
            searchSourceBuilder.size(2);//TODO: make this configurable
            SearchRequest searchRequest = new SearchRequest().source(searchSourceBuilder).indices(index);
            CountDownLatch latch = new CountDownLatch(1);
            LatchedActionListener listener = new LatchedActionListener<SearchResponse>(ActionListener.wrap(r -> {
                SearchHit[] hits = r.getHits().getHits();

                if (hits != null && hits.length > 0) {
                    StringBuilder contextBuilder = new StringBuilder();
                    for (int i = 0; i < hits.length; i++) {
                        SearchHit hit = hits[i];
                        Map<String, Object> sourceAsMap = hit.getSourceAsMap();
                        StringBuilder fieldContentBuilder = new StringBuilder();
                        AccessController.doPrivileged((PrivilegedExceptionAction<Void>) () -> {
                            for (String key : sourceAsMap.keySet()) {
                                Object content = sourceAsMap.get(key);
                                String contentStr = content instanceof String ? (String)content : gson.toJson(content);
                                fieldContentBuilder.append(key).append(": ").append(contentStr);
                            }
                            return null;
                        });

                        contextBuilder.append("document_id: ").append(hit.getId()).append("\\\\nDocument context:").append(fieldContentBuilder).append("\\\\n");
                    }
                    contextRef.set(gson.toJson(contextBuilder.toString()));
                }
            }, e -> {
                log.error("Failed to search index", e);
                exceptionRef.set(e);
            }), latch);
            client.search(searchRequest, listener);

            try {
                latch.await();
            } catch (InterruptedException e) {
                throw new IllegalStateException(e);
            }
            if (exceptionRef.get() != null) {
                throw new MLException(exceptionRef.get());
            }
            return (T)contextRef.get();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public String getDescription() {
        return description;
    }

    @Override
    public boolean validate(String input) {
        return parseInput(input) != null;
    }

    private Map<String, String> parseInput(String input) {
        Pattern pattern = Pattern.compile("\\S*Index:\\s*([\\w-]+),\\s*Query:\\s*(\\{.*\\})");
        Matcher matcher = pattern.matcher(input);
        if (matcher.find()) {
            String index = matcher.group(1);
            String query = matcher.group(2);
            if (index != null && query != null) {
                return ImmutableMap.of("index", index, "query", query);
            }
        }
        return null;
    }

    public void setClient(Client client) {
        this.client = client;
    }

    public void setXContentRegistry(NamedXContentRegistry xContentRegistry) {
        this.xContentRegistry = xContentRegistry;
    }
}
