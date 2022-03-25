/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine;

import com.amazon.randomcutforest.parkservices.state.ThresholdedRandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.opensearch.ml.common.Model;
import org.opensearch.ml.common.input.parameter.clustering.KMeansParams;
import org.opensearch.ml.common.input.parameter.regression.LinearRegressionParams;
import org.opensearch.ml.engine.algorithms.clustering.KMeans;
import org.opensearch.ml.engine.algorithms.regression.LinearRegression;
import org.opensearch.ml.engine.utils.ModelSerDeSer;
import org.tribuo.clustering.kmeans.KMeansModel;
import org.tribuo.regression.sgd.linear.LinearSGDModel;

import static org.junit.Assert.assertNotNull;
import static org.opensearch.ml.engine.helper.MLTestHelper.constructTestDataFrame;

public class ModelSerDeSerTest {
    @Rule
    public ExpectedException thrown = ExpectedException.none();

    private final RandomCutForestMapper rcfMapper = new RandomCutForestMapper();
    private final ThresholdedRandomCutForestMapper trcfMapper = new ThresholdedRandomCutForestMapper();

    @Test
    public void testModelSerDeSerKMeans() {
        KMeansParams params = KMeansParams.builder().build();
        KMeans kMeans = new KMeans(params);
        Model model = kMeans.train(constructTestDataFrame(100));

        KMeansModel deserializedModel = (KMeansModel) ModelSerDeSer.deserialize(model.getContent());
        assertNotNull(deserializedModel);
    }

    @Test
    public void testModelSerDeSerLinearRegression() {
        LinearRegressionParams params = LinearRegressionParams.builder().target("f2").build();
        LinearRegression linearRegression = new LinearRegression(params);
        Model model = linearRegression.train(constructTestDataFrame(100));

        LinearSGDModel deserializedModel = (LinearSGDModel) ModelSerDeSer.deserialize(model.getContent());
        assertNotNull(deserializedModel);
    }

}