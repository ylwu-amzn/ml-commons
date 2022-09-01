/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import java.io.IOException;
import org.junit.Rule;
import org.junit.rules.ExpectedException;
import org.opensearch.client.Response;
import org.opensearch.ml.utils.TestHelper;
import org.opensearch.rest.RestStatus;

public class RestMLUploadModelChunkActionIT extends MLCommonsRestTestCase {
    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();

    public void testUploadModelChunkAPI_Success() throws IOException {
        Response uploadModelChunkResponse = TestHelper
                .makeRequest(client(), "POST", "_plugins/_ml/upload_chunk/test_model/1/0/1", null, TestHelper.toHttpEntity("12345678"), null);
        assertNotNull(uploadModelChunkResponse);
        assertEquals(RestStatus.OK, TestHelper.restStatus(uploadModelChunkResponse));
    }
}
