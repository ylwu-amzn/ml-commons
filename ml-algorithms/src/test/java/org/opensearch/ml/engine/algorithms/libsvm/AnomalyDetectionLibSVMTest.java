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
import org.opensearch.ml.common.dataframe.ColumnMeta;
import org.opensearch.ml.common.dataframe.ColumnType;
import org.opensearch.ml.common.dataframe.ColumnValue;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataframe.DefaultDataFrame;
import org.opensearch.ml.common.dataframe.DoubleValue;
import org.opensearch.ml.common.dataframe.IntValue;
import org.opensearch.ml.common.dataframe.Row;
import org.opensearch.ml.engine.contants.TribuoOutputType;
import org.opensearch.ml.engine.utils.TribuoUtil;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.anomaly.AnomalyFactory;
import org.tribuo.anomaly.Event;
import org.tribuo.anomaly.example.AnomalyDataGenerator;
import org.tribuo.anomaly.libsvm.LibSVMAnomalyModel;
import org.tribuo.anomaly.libsvm.LibSVMAnomalyTrainer;
import org.tribuo.anomaly.libsvm.SVMAnomalyType;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.SVMParameters;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Optional;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class AnomalyDetectionLibSVMTest {

    @Test
    public void testExample() {
        Pair<Dataset<Event>,Dataset<Event>> pair = AnomalyDataGenerator.gaussianAnomaly(10,0.3);
        Dataset<Event> data = pair.getA();
        Dataset<Event> test = pair.getB();


        Iterator<Example<Event>> iterator = data.iterator();
        while(iterator.hasNext()) {
            Example<Event> next = iterator.next();
            Iterator<Feature> it1 = next.iterator();
            while(it1.hasNext()) {
                Feature f = it1.next();
                System.out.println(f.getName() + ": " + f.getValue());
            }

            Event output = next.getOutput();
            System.out.println(output.getScore() + ", " + output.getType());
        }
        System.out.println();

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

    @Test
    public void testExample2() {
        ColumnMeta[] columnMeta = new ColumnMeta[] {new ColumnMeta("A", ColumnType.DOUBLE),
                new ColumnMeta("B", ColumnType.DOUBLE),
                new ColumnMeta("anomaly_type", ColumnType.INTEGER)
        };
        List<Row> rows = new ArrayList<>();
        rows.add(new Row(new ColumnValue[]{new DoubleValue(1.1), new DoubleValue(1.2), new IntValue(1)}));
        rows.add(new Row(new ColumnValue[]{new DoubleValue(1.0), new DoubleValue(1.1), new IntValue(0)}));
        rows.add(new Row(new ColumnValue[]{new DoubleValue(1.2), new DoubleValue(1.2), new IntValue(-1)}));
        rows.add(new Row(new ColumnValue[]{new DoubleValue(1.15), new DoubleValue(1.2), new IntValue(1)}));
        rows.add(new Row(new ColumnValue[]{new DoubleValue(1.16), new DoubleValue(1.15), new IntValue(1)}));
        DataFrame dataFrame = new DefaultDataFrame(columnMeta, rows);

        MutableDataset<Event> examples = TribuoUtil.generateDataset(dataFrame, new AnomalyFactory(),
                "Anomaly detection LibSVM training data from OpenSearch", TribuoOutputType.ANOMALY_DETECTION_LIBSVM);
        System.out.println(examples);
    }
}
