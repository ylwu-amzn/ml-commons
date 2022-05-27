/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.stats;

import static org.opensearch.ml.utils.IntegTestUtils.TESTING_DATA;
import static org.opensearch.ml.utils.IntegTestUtils.generateMLTestingData;
import static org.opensearch.ml.utils.IntegTestUtils.verifyGeneratedTestingData;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import org.junit.Before;
import org.junit.Ignore;
import org.opensearch.action.ActionFuture;
import org.opensearch.ml.plugin.MachineLearningPlugin;
import org.opensearch.ml.stats.MLNodeLevelStat;
import org.opensearch.ml.stats.MLStatsInput;
import org.opensearch.plugins.Plugin;
import org.opensearch.test.OpenSearchIntegTestCase;

import com.google.common.collect.ImmutableSet;

public class MLStatsNodeITTests extends OpenSearchIntegTestCase {
    @Before
    public void initTestingData() throws ExecutionException, InterruptedException {
        generateMLTestingData();
    }

    @Override
    protected Collection<Class<? extends Plugin>> nodePlugins() {
        return Collections.singletonList(MachineLearningPlugin.class);
    }

    protected Collection<Class<? extends Plugin>> transportClientPlugins() {
        return Collections.singletonList(MachineLearningPlugin.class);
    }

    public void testGeneratedTestingData() throws ExecutionException, InterruptedException {
        verifyGeneratedTestingData(TESTING_DATA);
    }

    @Ignore
    public void testNormalCase() throws ExecutionException, InterruptedException {
        MLStatsNodesRequest request = new MLStatsNodesRequest(new String[0], new MLStatsInput());
        request.addNodeLevelStats(ImmutableSet.of(MLNodeLevelStat.ML_NODE_EXECUTING_TASK_COUNT));

        ActionFuture<MLStatsNodesResponse> future = client().execute(MLStatsNodesAction.INSTANCE, request);
        MLStatsNodesResponse response = future.get();
        assertNotNull(response);

        List<MLStatsNodeResponse> responseList = response.getNodes();
        // TODO: the responseList size here is not a fixed value. Comment out this assertion until this flaky test is fixed
        // assertEquals(1, responseList.size());
        assertNotNull(responseList);

        MLStatsNodeResponse nodeResponse = responseList.get(0);
        Map<MLNodeLevelStat, Object> statsMap = nodeResponse.getNodeStats();

        assertEquals(1, statsMap.size());
        assertEquals(0l, statsMap.get("ml_executing_task_count"));
    }
}
