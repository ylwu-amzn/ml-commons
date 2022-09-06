/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.custom;

import org.junit.Before;
import org.junit.Rule;
import org.junit.rules.ExpectedException;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.client.Client;
import org.opensearch.ml.action.custom.uploadchunk.MLModelChunkUploader;
import org.opensearch.ml.engine.algorithms.custom.CustomModelManager;
import org.opensearch.ml.indices.MLIndicesHandler;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.threadpool.ThreadPool;

public class MLModelChunkUploaderTests extends OpenSearchTestCase {
    @Mock
    CustomModelManager customModelManager;

    @Mock
    MLIndicesHandler mlIndicesHandler;

    @Mock
    MLTaskManager mlTaskManager;

    @Mock
    ThreadPool threadPool;

    @Mock
    Client client;

    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
    }

    public void testConstructor() {
        MLModelChunkUploader mlModelChunkUploader = new MLModelChunkUploader(
            customModelManager,
            mlIndicesHandler,
            mlTaskManager,
            threadPool,
            client
        );
        assertNotNull(mlModelChunkUploader);
    }
}
