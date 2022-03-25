/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.state.ThresholdedRandomCutForestMapper;
import com.amazon.randomcutforest.parkservices.state.ThresholdedRandomCutForestState;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.opensearch.ml.common.input.parameter.ad.BatchRCFParams;
import org.opensearch.ml.common.input.parameter.ad.FitRCFParams;
import org.opensearch.ml.common.input.parameter.clustering.KMeansParams;
import org.opensearch.ml.common.Model;
import org.opensearch.ml.common.input.parameter.regression.LinearRegressionParams;
import org.opensearch.ml.engine.algorithms.clustering.KMeans;
import org.opensearch.ml.engine.algorithms.rcf.BatchRandomCutForest;
import org.opensearch.ml.engine.algorithms.rcf.FixedInTimeRandomCutForest;
import org.opensearch.ml.engine.algorithms.regression.LinearRegression;
import org.opensearch.ml.engine.utils.ModelSerDeSer;
import org.tribuo.clustering.kmeans.KMeansModel;
import org.tribuo.regression.sgd.linear.LinearSGDModel;

import java.util.Arrays;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.opensearch.ml.engine.helper.MLTestHelper.TIME_FIELD;
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

        KMeansModel deserializedModel = ModelSerDeSer.deserialize(model.getContent(), KMeans.schema);
        byte[] serializedModel = ModelSerDeSer.serialize(deserializedModel, KMeans.schema);
        assertTrue(Arrays.equals(serializedModel, model.getContent()));
    }

    @Test
    public void testModelSerDeSerLinearRegression() {
        LinearRegressionParams params = LinearRegressionParams.builder().target("f2").build();
        LinearRegression linearRegression = new LinearRegression(params);
        Model model = linearRegression.train(constructTestDataFrame(100));

        LinearSGDModel deserializedModel = ModelSerDeSer.deserialize(model.getContent(), LinearRegression.schema);
        assertNotNull(deserializedModel);
    }

    @Test
    public void testModelSerDeSerBatchRCF() {
        BatchRCFParams params = BatchRCFParams.builder().build();
        BatchRandomCutForest batchRCF = new BatchRandomCutForest(params);
        Model model = batchRCF.train(constructTestDataFrame(500));

        RandomCutForestState deserializedState = ModelSerDeSer.deserialize(model.getContent(), BatchRandomCutForest.schema);
        RandomCutForest forest = rcfMapper.toModel(deserializedState);
        assertNotNull(forest);
        byte[] serializedModel = ModelSerDeSer.serialize(deserializedState, BatchRandomCutForest.schema);
        assertTrue(Arrays.equals(serializedModel, model.getContent()));
    }

    @Test
    public void testModelSerDeSerFitRCF() {
        FitRCFParams params = FitRCFParams.builder().timeField(TIME_FIELD).build();
        FixedInTimeRandomCutForest fitRCF = new FixedInTimeRandomCutForest(params);
        Model model = fitRCF.train(constructTestDataFrame(500, true));

        ThresholdedRandomCutForestState deserializedState = ModelSerDeSer.deserialize(model.getContent(), FixedInTimeRandomCutForest.schema);
        ThresholdedRandomCutForest forest = trcfMapper.toModel(deserializedState);
        assertNotNull(forest);
        byte[] serializedModel = ModelSerDeSer.serialize(deserializedState, FixedInTimeRandomCutForest.schema);
        assertTrue(Arrays.equals(serializedModel, model.getContent()));
    }

}