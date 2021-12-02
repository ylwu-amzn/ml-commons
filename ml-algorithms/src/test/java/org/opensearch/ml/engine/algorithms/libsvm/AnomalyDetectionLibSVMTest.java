/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 *
 */

package org.opensearch.ml.engine.algorithms.libsvm;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.Test;
import org.tribuo.Dataset;
import org.tribuo.Prediction;
import org.tribuo.anomaly.Event;
import org.tribuo.anomaly.example.AnomalyDataGenerator;
import org.tribuo.anomaly.libsvm.LibSVMAnomalyModel;
import org.tribuo.anomaly.libsvm.LibSVMAnomalyTrainer;
import org.tribuo.anomaly.libsvm.SVMAnomalyType;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.SVMParameters;

import java.util.List;
import java.util.Optional;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class AnomalyDetectionLibSVMTest {

    @Test
    public void testExample() {
        Pair<Dataset<Event>,Dataset<Event>> pair = AnomalyDataGenerator.gaussianAnomaly(1000,0.3);
        Dataset<Event> data = pair.getA();
        Dataset<Event> test = pair.getB();

        SVMParameters params = new SVMParameters<>(new SVMAnomalyType(SVMAnomalyType.SVMMode.ONE_CLASS), KernelType.RBF);
        params.setGamma(1.0);
        params.setNu(0.1);

        LibSVMAnomalyTrainer trainer = new LibSVMAnomalyTrainer(params);
        LibSVMModel model = trainer.train(data);
        ((LibSVMAnomalyModel)model).getNumberOfSupportVectors();

        List<Prediction<Event>> predictions = model.predict(test);
        assertEquals(1000, predictions.size());
        Optional<Prediction<Event>> anomaly = predictions.stream().filter(p -> p.getOutput().getType() == Event.EventType.ANOMALOUS).findAny();
        assertTrue(anomaly.isPresent());
    }

}
