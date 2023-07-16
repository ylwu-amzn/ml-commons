/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.opensearch.ml.utils.TestHelper.getCreateConnectorRestRequest;
import static org.opensearch.ml.utils.TestHelper.verifyParsedCreateConnectorInput;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Before;
import org.junit.Rule;
import org.junit.rules.ExpectedException;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.opensearch.action.ActionListener;
import org.opensearch.client.node.NodeClient;
import org.opensearch.common.Strings;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.transport.connector.MLCreateConnectorAction;
import org.opensearch.ml.common.transport.connector.MLCreateConnectorInput;
import org.opensearch.ml.common.transport.connector.MLCreateConnectorRequest;
import org.opensearch.ml.common.transport.connector.MLCreateConnectorResponse;
import org.opensearch.rest.RestChannel;
import org.opensearch.rest.RestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.test.rest.FakeRestRequest;
import org.opensearch.threadpool.TestThreadPool;
import org.opensearch.threadpool.ThreadPool;

public class RestMLCreateConnectorActionTests extends OpenSearchTestCase {
    @Rule
    public ExpectedException thrown = ExpectedException.none();

    private RestMLCreateConnectorAction restMLCreateConnectorAction;

    NodeClient client;
    private ThreadPool threadPool;

    @Mock
    RestChannel channel;

    @Before
    public void setup() {
        restMLCreateConnectorAction = new RestMLCreateConnectorAction();

        threadPool = new TestThreadPool(this.getClass().getSimpleName() + "ThreadPool");
        client = spy(new NodeClient(Settings.EMPTY, threadPool));

        doAnswer(invocation -> {
            ActionListener<MLCreateConnectorResponse> actionListener = invocation.getArgument(2);
            return null;
        }).when(client).execute(eq(MLCreateConnectorAction.INSTANCE), any(), any());

    }

    @Override
    public void tearDown() throws Exception {
        super.tearDown();
        threadPool.shutdown();
        client.close();
    }

    public void testConstructor() {
        RestMLCreateConnectorAction mlCreateConnectorAction = new RestMLCreateConnectorAction();
        assertNotNull(mlCreateConnectorAction);
    }

    public void testGetName() {
        String actionName = restMLCreateConnectorAction.getName();
        assertFalse(Strings.isNullOrEmpty(actionName));
        assertEquals("ml_create_connector_action", actionName);
    }

    public void testRoutes() {
        List<RestHandler.Route> routes = restMLCreateConnectorAction.routes();
        assertNotNull(routes);
        assertFalse(routes.isEmpty());
        RestHandler.Route route = routes.get(0);
        assertEquals(RestRequest.Method.POST, route.getMethod());
        assertEquals("/_plugins/_ml/connectors/_create", route.getPath());
    }

    public void testGetRequest() throws IOException {
        RestRequest request = getCreateConnectorRestRequest();
        MLCreateConnectorRequest mlCreateConnectorRequest = restMLCreateConnectorAction.getRequest(request);

        MLCreateConnectorInput mlCreateConnectorInput = mlCreateConnectorRequest.getMlCreateConnectorInput();
        verifyParsedCreateConnectorInput(mlCreateConnectorInput);
    }

    public void testPrepareRequest() throws Exception {
        RestRequest request = getCreateConnectorRestRequest();
        restMLCreateConnectorAction.handleRequest(request, channel, client);

        ArgumentCaptor<MLCreateConnectorRequest> argumentCaptor = ArgumentCaptor.forClass(MLCreateConnectorRequest.class);
        verify(client, times(1)).execute(eq(MLCreateConnectorAction.INSTANCE), argumentCaptor.capture(), any());
        MLCreateConnectorInput mlCreateConnectorInput = argumentCaptor.getValue().getMlCreateConnectorInput();
        verifyParsedCreateConnectorInput(mlCreateConnectorInput);
    }

    public void testPrepareRequest_EmptyContent() throws Exception {
        thrown.expect(IOException.class);
        thrown.expectMessage("Create Connector request has empty body");
        Map<String, String> params = new HashMap<>();
        RestRequest request = new FakeRestRequest.Builder(NamedXContentRegistry.EMPTY).withParams(params).build();

        restMLCreateConnectorAction.handleRequest(request, channel, client);
    }
}
