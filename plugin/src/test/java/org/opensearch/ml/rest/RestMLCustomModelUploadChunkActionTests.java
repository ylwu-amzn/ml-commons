/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;
import static org.mockito.Mockito.times;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Before;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.opensearch.action.ActionListener;
import org.opensearch.client.node.NodeClient;
import org.opensearch.common.Strings;
import org.opensearch.common.bytes.BytesArray;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.transport.custom.uploadchunk.MLUploadChunkInput;
import org.opensearch.ml.common.transport.custom.uploadchunk.MLUploadModelChunkAction;
import org.opensearch.ml.common.transport.custom.uploadchunk.MLUploadModelChunkRequest;
import org.opensearch.ml.common.transport.model.MLModelGetResponse;
import org.opensearch.rest.RestChannel;
import org.opensearch.rest.RestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.test.rest.FakeRestRequest;
import org.opensearch.threadpool.TestThreadPool;
import org.opensearch.threadpool.ThreadPool;

public class RestMLCustomModelUploadChunkActionTests extends OpenSearchTestCase {

    private RestMLCustomModelUploadChunkAction restMLCustomModelUploadChunkAction;
    NodeClient client;
    private ThreadPool threadPool;

    @Mock
    RestChannel channel;

    @Before
    public void setup() {
        restMLCustomModelUploadChunkAction = new RestMLCustomModelUploadChunkAction();

        threadPool = new TestThreadPool(this.getClass().getSimpleName() + "ThreadPool");
        client = spy(new NodeClient(Settings.EMPTY, threadPool));

        doAnswer(invocation -> {
            ActionListener<MLModelGetResponse> actionListener = invocation.getArgument(2);
            return null;
        }).when(client).execute(eq(MLUploadModelChunkAction.INSTANCE), any(), any());
    }

    @Override
    public void tearDown() throws Exception {
        super.tearDown();
        threadPool.shutdown();
        client.close();
    }

    public void testConstructor() {
        RestMLCustomModelUploadChunkAction mlUploadChunk = new RestMLCustomModelUploadChunkAction();
        assertNotNull(mlUploadChunk);
    }

    public void testGetName() {
        String actionName = restMLCustomModelUploadChunkAction.getName();
        assertFalse(Strings.isNullOrEmpty(actionName));
        assertEquals("ml_upload_model_chunk_action", actionName);
    }

    public void testRoutes() {
        List<RestHandler.Route> routes = restMLCustomModelUploadChunkAction.routes();
        assertNotNull(routes);
        assertFalse(routes.isEmpty());
        RestHandler.Route route = routes.get(0);
        assertEquals(RestRequest.Method.POST, route.getMethod());
        assertEquals("/_plugins/_ml/custom_model/upload_chunk/{name}/{version}/{chunk_number}/{total_chunks}", route.getPath());
    }

    public void test_PrepareRequest() throws Exception {
        RestRequest request = getRestRequest();
        restMLCustomModelUploadChunkAction.handleRequest(request, channel, client);

        ArgumentCaptor<MLUploadModelChunkRequest> argumentCaptor = ArgumentCaptor.forClass(MLUploadModelChunkRequest.class);
        verify(client, times(1)).execute(eq(MLUploadModelChunkAction.INSTANCE), argumentCaptor.capture(), any());
        MLUploadChunkInput uploadChunkInput = argumentCaptor.getValue().getMlUploadInput();
        assertEquals("test_model", uploadChunkInput.getName());
        assertEquals(Integer.valueOf(1), uploadChunkInput.getVersion());
        assertEquals(Integer.valueOf(0), uploadChunkInput.getChunkNumber());
        assertEquals(Integer.valueOf(1), uploadChunkInput.getTotalChunks());
        assertNotNull(uploadChunkInput.getContent());
    }

    private RestRequest getRestRequest() {
        RestRequest.Method method = RestRequest.Method.POST;
        BytesArray content = new BytesArray("12345678");
        Map<String, String> params = new HashMap<>();
        params.put("name", "test_model");
        params.put("version", "1");
        params.put("chunk_number", "0");
        params.put("total_chunks", "1");
        RestRequest request = new FakeRestRequest.Builder(NamedXContentRegistry.EMPTY)
            .withMethod(method)
            .withParams(params)
            .withContent(content, null)
            .build();
        return request;
    }
}
