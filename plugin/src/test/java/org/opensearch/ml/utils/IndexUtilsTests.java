/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.utils;

import static org.mockito.Mockito.mock;

import org.junit.Before;
import org.opensearch.action.ActionListener;
import org.opensearch.action.support.master.AcknowledgedResponse;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.test.OpenSearchIntegTestCase;

import com.google.common.collect.ImmutableMap;

public class IndexUtilsTests extends OpenSearchIntegTestCase {
    private IndexNameExpressionResolver indexNameResolver;

    @Before
    public void setup() {
        indexNameResolver = mock(IndexNameExpressionResolver.class);
    }

    public void testGetIndexHealth_NoIndex() {
        IndexUtils indexUtils = new IndexUtils(client(), clusterService(), indexNameResolver);
        String output = indexUtils.getIndexHealthStatus("test");
        assertEquals(IndexUtils.NONEXISTENT_INDEX_STATUS, output);
    }

    public void testGetIndexHealth_Index() {
        String indexName = "test-2";
        createIndex(indexName);
        flush();
        IndexUtils indexUtils = new IndexUtils(client(), clusterService(), indexNameResolver);
        String status = indexUtils.getIndexHealthStatus(indexName);
        assertTrue(status.equals("green") || status.equals("yellow"));
    }

    public void testGetIndexHealth_Alias() {
        String indexName = "test-2";
        String aliasName = "alias";
        createIndex(indexName);
        flush();
        AcknowledgedResponse response = client().admin().indices().prepareAliases().addAlias(indexName, aliasName).execute().actionGet();
        assertTrue(response.isAcknowledged());
        IndexUtils indexUtils = new IndexUtils(client(), clusterService(), indexNameResolver);
        String status = indexUtils.getIndexHealthStatus(aliasName);
        assertTrue(status.equals("green") || status.equals("yellow"));
    }

    public void testGetNumberOfDocumentsInIndex_NonExistentIndex() {
        IndexUtils indexUtils = new IndexUtils(client(), clusterService(), indexNameResolver);
        indexUtils.getNumberOfDocumentsInIndex("index", ActionListener.wrap(r -> { assertEquals((Long) 0L, r); }, e -> { assertNull(e); }));
    }

    public void testGetNumberOfDocumentsInIndex_RegularIndex() {
        String indexName = "test-2";
        createIndex(indexName);
        flush();

        long count = 20;
        for (int i = 0; i < count; i++) {
            index(indexName, "_doc", i + "", ImmutableMap.of(randomAlphaOfLength(5), randomAlphaOfLength(5)));
        }
        flushAndRefresh(indexName);
        IndexUtils indexUtils = new IndexUtils(client(), clusterService(), indexNameResolver);
        indexUtils
            .getNumberOfDocumentsInIndex(indexName, ActionListener.wrap(r -> { assertEquals((Long) count, r); }, e -> { assertNull(e); }));
    }
}
