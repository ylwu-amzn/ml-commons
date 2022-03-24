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
import org.opensearch.ml.engine.algorithms.clustering.KMeans;
import org.opensearch.ml.engine.algorithms.rcf.BatchRandomCutForest;
import org.opensearch.ml.engine.algorithms.rcf.FixedInTimeRandomCutForest;
import org.opensearch.ml.engine.exceptions.ModelSerDeSerException;
import org.opensearch.ml.engine.utils.ModelSerDeSer;
import org.tribuo.clustering.kmeans.KMeansModel;

import java.util.Arrays;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.opensearch.ml.engine.helper.MLTestHelper.TIME_FIELD;
import static org.opensearch.ml.engine.helper.MLTestHelper.constructTestDataFrame;

public class ModelSerDeSerTest {
    @Rule
    public ExpectedException thrown = ExpectedException.none();

    private final Object dummyModel = new Object();
    private final RandomCutForestMapper rcfMapper = new RandomCutForestMapper();
    private final ThresholdedRandomCutForestMapper trcfMapper = new ThresholdedRandomCutForestMapper();

    @Test
    public void testModelSerDeSerBlocklModel() {
        thrown.expect(ModelSerDeSerException.class);
        byte[] modelBin = ModelSerDeSer.serialize(dummyModel);
        Object model = ModelSerDeSer.deserialize(modelBin);
        assertTrue(model.equals(dummyModel));
    }

    @Test
    public void testModelSerDeSerKMeans() {
        KMeansParams params = KMeansParams.builder().build();
        KMeans kMeans = new KMeans(params);
        Model model = kMeans.train(constructTestDataFrame(100));

        KMeansModel kMeansModel = (KMeansModel) ModelSerDeSer.deserialize(model.getContent());
        byte[] serializedModel = ModelSerDeSer.serialize(kMeansModel);
        assertFalse(Arrays.equals(serializedModel, model.getContent()));
    }

    @Test
    public void testModelSerDeSerBatchRCF() {
        BatchRCFParams params = BatchRCFParams.builder().build();
        BatchRandomCutForest batchRCF = new BatchRandomCutForest(params);
        Model model = batchRCF.train(constructTestDataFrame(500));

        RandomCutForestState state = (RandomCutForestState) ModelSerDeSer.deserialize(model.getContent());
        RandomCutForest forest = rcfMapper.toModel(state);
        assertNotNull(forest);
        byte[] serializedModel = ModelSerDeSer.serialize(ModelSerDeSer.serialize(state));
        assertFalse(Arrays.equals(serializedModel, model.getContent()));
    }

    @Test
    public void testModelSerDeSerFitRCF() {
        FitRCFParams params = FitRCFParams.builder().timeField(TIME_FIELD).build();
        FixedInTimeRandomCutForest fitRCF = new FixedInTimeRandomCutForest(params);
        Model model = fitRCF.train(constructTestDataFrame(500, true));

        ThresholdedRandomCutForestState state = (ThresholdedRandomCutForestState) ModelSerDeSer.deserialize(model.getContent());
        ThresholdedRandomCutForest forest = trcfMapper.toModel(state);
        assertNotNull(forest);
        byte[] serializedModel = ModelSerDeSer.serialize(ModelSerDeSer.serialize(state));
        assertFalse(Arrays.equals(serializedModel, model.getContent()));
    }

}