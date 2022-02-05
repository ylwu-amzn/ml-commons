/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine;

import com.amazon.randomcutforest.RandomCutForest;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataset.DataFrameInputDataset;
import org.opensearch.ml.common.dataset.MLInputDataset;
import org.opensearch.ml.common.parameter.FunctionName;
import org.opensearch.ml.common.parameter.Input;
import org.opensearch.ml.common.parameter.KMeansParams;
import org.opensearch.ml.common.parameter.LinearRegressionParams;
import org.opensearch.ml.common.parameter.MLInput;
import org.opensearch.ml.common.parameter.Model;

import java.util.ArrayList;
import java.util.List;

import static org.opensearch.ml.engine.helper.KMeansHelper.constructKMeansDataFrame;
import static org.opensearch.ml.engine.helper.LinearRegressionHelper.constructLinearRegressionTrainDataFrame;

public class RCFTest {
    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();

    List<double[]> fourclassExpectedData;
    List<double[]> fourclassAnomalyData;
    int trainingDataSize = 100;
    String index_name = "test_data_ad_predict_result_rcf9";

    @Before
    public void setUp() {
        fourclassExpectedData = createExpectedData();
        fourclassAnomalyData = createAnomalyData();
    }

    @Test
    public void testRCF_TrainWithExpectedDataOnly() {
        trainingDataSize = 200;
        RandomCutForest forest = RandomCutForest.builder()
                .numberOfTrees(10)
                .sampleSize(trainingDataSize)
                .outputAfter(trainingDataSize)
                .dimensions(2) // still required!
                .randomSeed(123)
                .storeSequenceIndexesEnabled(true)
                .centerOfMassEnabled(true)
                .build();

        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        int i;
        for (i = 0; i < trainingDataSize; i++) {
            double[] point = fourclassExpectedData.get(i);
            double score = forest.getAnomalyScore(point);
            if (score < min) {
                min = score;
            }
            if (score > max) {
                max = score;
            }
            forest.update(point);
        }

        max = Double.MIN_VALUE;
        min = Double.MAX_VALUE;
        int j;
        for (j = 0; j < fourclassAnomalyData.size(); j++) {
            double[] point = fourclassAnomalyData.get(j);
            double score = forest.getAnomalyScore(point);
            if (score < min) {
                min = score;
            }
            if (score > max) {
                max = score;
            }
            System.out.println(String.format("{ \"index\" : { \"_index\" : \"%s\", \"_id\" : \"%s\" } }",
                    index_name, j));
            System.out.println(String.format("{\"A\":%s,\"B\":%s,\"score\":%s,\"anomaly_type\":\"%s\"}",
                    point[0], point[1], score, score > 1 ? "ANOMALOUS" : "EXPECTED"));
        }

        for (int m = trainingDataSize; m < fourclassExpectedData.size(); m++) {
            double[] point = fourclassExpectedData.get(m);
            double score = forest.getAnomalyScore(point);
            if (score < min) {
                min = score;
            }
            if (score > max) {
                max = score;
            }
            System.out.println(String.format("{ \"index\" : { \"_index\" : \"%s\", \"_id\" : \"%s\" } }",
                    index_name, j + m));
            System.out.println(String.format("{\"A\":%s,\"B\":%s,\"score\":%s,\"anomaly_type\":\"%s\"}",
                    point[0], point[1], score, score > 1 ? "ANOMALOUS" : "EXPECTED"));
        }
    }

    @Test
    public void testRCF_TrainWihExpectedAndAnomalyData() {
        trainingDataSize = 200;
        int expectedTrainingDataSize = 198;
        int anomalyTrainingDataSize = 2;
        RandomCutForest forest = RandomCutForest.builder()
                .numberOfTrees(10)
                .sampleSize(trainingDataSize)
                .outputAfter(trainingDataSize)
                .dimensions(2) // still required!
                .randomSeed(123)
                .storeSequenceIndexesEnabled(true)
                .centerOfMassEnabled(true)
                .build();

        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        // train with normal data
        for (int i = 0; i < expectedTrainingDataSize; i++) {
            double[] point = fourclassExpectedData.get(i);
            double score = forest.getAnomalyScore(point);
            if (score < min) {
                min = score;
            }
            if (score > max) {
                max = score;
            }
            forest.update(point);
        }

        // train with anomaly data
        for (int i = 0; i < anomalyTrainingDataSize; i++) {
            double[] point = fourclassAnomalyData.get(i);
            double score = forest.getAnomalyScore(point);
            if (score < min) {
                min = score;
            }
            if (score > max) {
                max = score;
            }
            forest.update(point);
        }

        max = Double.MIN_VALUE;
        min = Double.MAX_VALUE;
        int j;
        // Predict anomaly data
        for (j = anomalyTrainingDataSize; j < fourclassAnomalyData.size(); j++) {
            double[] point = fourclassAnomalyData.get(j);
            double score = forest.getAnomalyScore(point);
            if (score < min) {
                min = score;
            }
            if (score > max) {
                max = score;
            }
            System.out.println(String.format("{ \"index\" : { \"_index\" : \"%s\", \"_id\" : \"%s\" } }",
                    index_name, j));
            System.out.println(String.format("{\"A\":%s,\"B\":%s,\"score\":%s,\"anomaly_type\":\"%s\"}",
                    point[0], point[1], score, score > 1 ? "ANOMALOUS" : "EXPECTED"));
        }
        // Predict normal data
        for (int m = expectedTrainingDataSize; m < fourclassExpectedData.size(); m++) {
            double[] point = fourclassExpectedData.get(m);
            double score = forest.getAnomalyScore(point);
            if (score < min) {
                min = score;
            }
            if (score > max) {
                max = score;
            }
            System.out.println(String.format("{ \"index\" : { \"_index\" : \"%s\", \"_id\" : \"%s\" } }",
                    index_name, j + m));
            System.out.println(String.format("{\"A\":%s,\"B\":%s,\"score\":%s,\"anomaly_type\":\"%s\"}",
                    point[0], point[1], score, score > 1 ? "ANOMALOUS" : "EXPECTED"));
        }
    }


    public List<double[]> createExpectedData() {
        List<double[]> fourclassExpectedData = new ArrayList<>();
        fourclassExpectedData.add(new double[]{0.747253, 0.894737});
        fourclassExpectedData.add(new double[]{0.472527, -0.0643275});
        fourclassExpectedData.add(new double[]{0.362637, 0.789474});
        fourclassExpectedData.add(new double[]{-0.692308, 0.836257});
        fourclassExpectedData.add(new double[]{0.153846, 0.274854});
        fourclassExpectedData.add(new double[]{0.10989, -0.929825});
        fourclassExpectedData.add(new double[]{-0.032967, -0.00584795});
        fourclassExpectedData.add(new double[]{-0.747253, -0.602339});
        fourclassExpectedData.add(new double[]{-0.582418, 0.847953});
        fourclassExpectedData.add(new double[]{-0.351648, -0.602339});
        fourclassExpectedData.add(new double[]{-0.725275, -0.438596});
        fourclassExpectedData.add(new double[]{-0.824176, -0.48538});
        fourclassExpectedData.add(new double[]{-0.10989, 0.380117});
        fourclassExpectedData.add(new double[]{-0.373626, 0.450292});
        fourclassExpectedData.add(new double[]{0.934066, 0.555556});
        fourclassExpectedData.add(new double[]{0.417582, 0.80117});
        fourclassExpectedData.add(new double[]{-0.186813, -0.00584795});
        fourclassExpectedData.add(new double[]{-0.296703, 0.497076});
        fourclassExpectedData.add(new double[]{0.153846, -0.0994152});
        fourclassExpectedData.add(new double[]{0.450549, -0.0760234});
        fourclassExpectedData.add(new double[]{-0.769231, -0.415205});
        fourclassExpectedData.add(new double[]{-0.010989, -0.0292398});
        fourclassExpectedData.add(new double[]{-0.142857, -0.578947});
        fourclassExpectedData.add(new double[]{-0.0879121, -0.871345});
        fourclassExpectedData.add(new double[]{-0.032967, 0.368421});
        fourclassExpectedData.add(new double[]{0.758242, 0.859649});
        fourclassExpectedData.add(new double[]{0.736264, 0.80117});
        fourclassExpectedData.add(new double[]{-0.571429, 0.22807});
        fourclassExpectedData.add(new double[]{0.351648, 0.169591});
        fourclassExpectedData.add(new double[]{0.846154, 0.836257});
        fourclassExpectedData.add(new double[]{-0.648352, 0.695906});
        fourclassExpectedData.add(new double[]{0.186813, 0.309942});
        fourclassExpectedData.add(new double[]{-0.901099, 0.239766});
        fourclassExpectedData.add(new double[]{-0.406593, -0.0994152});
        fourclassExpectedData.add(new double[]{-0.637363, 0.0877193});
        fourclassExpectedData.add(new double[]{0.21978, -0.812865});
        fourclassExpectedData.add(new double[]{-0.78022, -0.461988});
        fourclassExpectedData.add(new double[]{0.395604, 0.0175439});
        fourclassExpectedData.add(new double[]{-0.956044, 0.473684});
        fourclassExpectedData.add(new double[]{0.296703, 0.216374});
        fourclassExpectedData.add(new double[]{0.10989, -0.707602});
        fourclassExpectedData.add(new double[]{-0.373626, 0.590643});
        fourclassExpectedData.add(new double[]{0.0659341, -0.730994});
        fourclassExpectedData.add(new double[]{-0.56044, 0.660819});
        fourclassExpectedData.add(new double[]{-0.813187, 0.356725});
        fourclassExpectedData.add(new double[]{0.142857, -0.836257});
        fourclassExpectedData.add(new double[]{-0.32967, 0.578947});
        fourclassExpectedData.add(new double[]{-0.989011, 0.181287});
        fourclassExpectedData.add(new double[]{-0.648352, -0.157895});
        fourclassExpectedData.add(new double[]{-0.296703, -0.567251});
        fourclassExpectedData.add(new double[]{0.384615, -0.157895});
        fourclassExpectedData.add(new double[]{-0.208791, -0.777778});
        fourclassExpectedData.add(new double[]{-0.395604, 0.578947});
        fourclassExpectedData.add(new double[]{0.835165, 0.660819});
        fourclassExpectedData.add(new double[]{-0.637363, 0.111111});
        fourclassExpectedData.add(new double[]{-0.549451, 0.719298});
        fourclassExpectedData.add(new double[]{0.043956, -0.777778});
        fourclassExpectedData.add(new double[]{-0.010989, -0.672515});
        fourclassExpectedData.add(new double[]{0.021978, -0.0760234});
        fourclassExpectedData.add(new double[]{0.384615, 0.789474});
        fourclassExpectedData.add(new double[]{-0.868132, 0.438596});
        fourclassExpectedData.add(new double[]{-0.186813, -0.730994});
        fourclassExpectedData.add(new double[]{-0.373626, 0.146199});
        fourclassExpectedData.add(new double[]{-0.527473, 0.80117});
        fourclassExpectedData.add(new double[]{-0.32967, -0.637427});
        fourclassExpectedData.add(new double[]{0.417582, -0.192982});
        fourclassExpectedData.add(new double[]{0.153846, 0.00584795});
        fourclassExpectedData.add(new double[]{0.241758, -0.824561});
        fourclassExpectedData.add(new double[]{-0.230769, 0.380117});
        fourclassExpectedData.add(new double[]{-0.10989, -0.847953});
        fourclassExpectedData.add(new double[]{-0.659341, -0.0409357});
        fourclassExpectedData.add(new double[]{0.868132, 0.695906});
        fourclassExpectedData.add(new double[]{0.0989011, -0.0877193});
        fourclassExpectedData.add(new double[]{-0.714286, 0.122807});
        fourclassExpectedData.add(new double[]{0.252747, 0.356725});
        fourclassExpectedData.add(new double[]{-0.846154, 0.380117});
        fourclassExpectedData.add(new double[]{-0.406593, 0.134503});
        fourclassExpectedData.add(new double[]{-0.56044, -0.111111});
        fourclassExpectedData.add(new double[]{-0.318681, 0.473684});
        fourclassExpectedData.add(new double[]{-0.648352, 0.707602});
        fourclassExpectedData.add(new double[]{-0.516484, -0.812865});
        fourclassExpectedData.add(new double[]{0.0659341, 0.28655});
        fourclassExpectedData.add(new double[]{0.791209, 0.918129});
        fourclassExpectedData.add(new double[]{0.934066, 0.602339});
        fourclassExpectedData.add(new double[]{-0.802198, -0.602339});
        fourclassExpectedData.add(new double[]{0.791209, 0.871345});
        fourclassExpectedData.add(new double[]{0.505495, 0.929825});
        fourclassExpectedData.add(new double[]{-0.472527, -0.74269});
        fourclassExpectedData.add(new double[]{-0.142857, -0.707602});
        fourclassExpectedData.add(new double[]{0.362637, -0.239766});
        fourclassExpectedData.add(new double[]{-0.582418, 0.239766});
        fourclassExpectedData.add(new double[]{0.21978, -0.964912});
        fourclassExpectedData.add(new double[]{-0.362637, 0.403509});
        fourclassExpectedData.add(new double[]{-0.637363, -0.625731});
        fourclassExpectedData.add(new double[]{0.351648, 0.146199});
        fourclassExpectedData.add(new double[]{-0.021978, 0.426901});
        fourclassExpectedData.add(new double[]{-0.230769, 0.508772});
        fourclassExpectedData.add(new double[]{-0.758242, -0.438596});
        fourclassExpectedData.add(new double[]{-0.505495, -0.836257});
        fourclassExpectedData.add(new double[]{0.373626, -0.0760234});
        fourclassExpectedData.add(new double[]{0.417582, 0.0643275});
        fourclassExpectedData.add(new double[]{-0.78022, -0.181287});
        fourclassExpectedData.add(new double[]{0.0989011, -0.836257});
        fourclassExpectedData.add(new double[]{-0.714286, 0.906433});
        fourclassExpectedData.add(new double[]{-0.945055, 0.649123});
        fourclassExpectedData.add(new double[]{0.703297, 0.812865});
        fourclassExpectedData.add(new double[]{-0.571429, -0.0760234});
        fourclassExpectedData.add(new double[]{-0.208791, 0.48538});
        fourclassExpectedData.add(new double[]{-0.879121, 0.403509});
        fourclassExpectedData.add(new double[]{0.868132, 0.614035});
        fourclassExpectedData.add(new double[]{-0.0879121, -0.695906});
        fourclassExpectedData.add(new double[]{-0.406593, 0.532164});
        fourclassExpectedData.add(new double[]{-0.142857, -0.122807});
        fourclassExpectedData.add(new double[]{-0.142857, -0.0526316});
        fourclassExpectedData.add(new double[]{-0.538462, 0.111111});
        fourclassExpectedData.add(new double[]{0.208791, 0.380117});
        fourclassExpectedData.add(new double[]{-0.615385, -0.134503});
        fourclassExpectedData.add(new double[]{-0.21978, -0.122807});
        fourclassExpectedData.add(new double[]{-0.736264, -0.0877193});
        fourclassExpectedData.add(new double[]{-0.461538, -0.0409357});
        fourclassExpectedData.add(new double[]{0.769231, 0.789474});
        fourclassExpectedData.add(new double[]{0.0659341, -0.94152});
        fourclassExpectedData.add(new double[]{-0.802198, 0.906433});
        fourclassExpectedData.add(new double[]{0.384615, 0.94152});
        fourclassExpectedData.add(new double[]{-0.252747, 0.321637});
        fourclassExpectedData.add(new double[]{-0.747253, -0.532164});
        fourclassExpectedData.add(new double[]{-0.21978, 0.309942});
        fourclassExpectedData.add(new double[]{0.516484, -0.146199});
        fourclassExpectedData.add(new double[]{-0.274725, -0.766082});
        fourclassExpectedData.add(new double[]{-0.604396, 0.847953});
        fourclassExpectedData.add(new double[]{-0.681319, 0.637427});
        fourclassExpectedData.add(new double[]{0.527473, 0.964912});
        fourclassExpectedData.add(new double[]{-0.417582, -0.660819});
        fourclassExpectedData.add(new double[]{-0.538462, 0.22807});
        fourclassExpectedData.add(new double[]{-0.582418, 0.181287});
        fourclassExpectedData.add(new double[]{-0.703297, -0.74269});
        fourclassExpectedData.add(new double[]{0.032967, -0.134503});
        fourclassExpectedData.add(new double[]{-0.351648, -0.567251});
        fourclassExpectedData.add(new double[]{-0.626374, -0.707602});
        fourclassExpectedData.add(new double[]{-0.318681, -0.719298});
        fourclassExpectedData.add(new double[]{-0.692308, -0.754386});
        fourclassExpectedData.add(new double[]{-0.538462, -0.812865});
        fourclassExpectedData.add(new double[]{-0.626374, 0.707602});
        fourclassExpectedData.add(new double[]{-0.67033, 0.789474});
        fourclassExpectedData.add(new double[]{-0.0879121, 0.0643275});
        fourclassExpectedData.add(new double[]{-0.747253, 0.111111});
        fourclassExpectedData.add(new double[]{-0.604396, 0.94152});
        fourclassExpectedData.add(new double[]{-0.384615, -0.660819});
        fourclassExpectedData.add(new double[]{-0.164835, -0.730994});
        fourclassExpectedData.add(new double[]{-0.340659, -0.730994});
        fourclassExpectedData.add(new double[]{-0.384615, 0.48538});
        fourclassExpectedData.add(new double[]{0.340659, 0.929825});
        fourclassExpectedData.add(new double[]{0.67033, 0.847953});
        fourclassExpectedData.add(new double[]{-0.351648, -0.508772});
        fourclassExpectedData.add(new double[]{0.0769231, 0.274854});
        fourclassExpectedData.add(new double[]{-0.835165, -0.54386});
        fourclassExpectedData.add(new double[]{0.032967, -0.94152});
        fourclassExpectedData.add(new double[]{0.0769231, -0.660819});
        fourclassExpectedData.add(new double[]{-0.406593, -0.730994});
        fourclassExpectedData.add(new double[]{-0.318681, -0.54386});
        fourclassExpectedData.add(new double[]{-0.824176, -0.520468});
        fourclassExpectedData.add(new double[]{-0.307692, -0.695906});
        fourclassExpectedData.add(new double[]{-0.285714, 0.567251});
        fourclassExpectedData.add(new double[]{0.0549451, 0.415205});
        fourclassExpectedData.add(new double[]{0.582418, 0.976608});
        fourclassExpectedData.add(new double[]{-0.527473, -0.730994});
        fourclassExpectedData.add(new double[]{-0.89011, 0.695906});
        fourclassExpectedData.add(new double[]{0.043956, -0.918129});
        fourclassExpectedData.add(new double[]{-0.021978, -0.122807});
        fourclassExpectedData.add(new double[]{0.021978, -0.204678});
        fourclassExpectedData.add(new double[]{0.483516, -0.0877193});
        fourclassExpectedData.add(new double[]{-0.747253, 0.0643275});
        fourclassExpectedData.add(new double[]{-1.0, 0.321637});
        fourclassExpectedData.add(new double[]{-0.263736, -0.157895});
        fourclassExpectedData.add(new double[]{0.549451, 0.883041});
        fourclassExpectedData.add(new double[]{0.648352, 0.812865});
        fourclassExpectedData.add(new double[]{-0.32967, -0.146199});
        fourclassExpectedData.add(new double[]{-0.472527, -0.754386});
        fourclassExpectedData.add(new double[]{-0.703297, 0.0760234});
        fourclassExpectedData.add(new double[]{0.0769231, 0.00584795});
        fourclassExpectedData.add(new double[]{0.362637, 0.812865});
        fourclassExpectedData.add(new double[]{-0.604396, 0.602339});
        fourclassExpectedData.add(new double[]{-0.351648, -0.122807});
        fourclassExpectedData.add(new double[]{0.527473, -0.111111});
        fourclassExpectedData.add(new double[]{0.494505, 0.859649});
        fourclassExpectedData.add(new double[]{-0.186813, -0.0526316});
        fourclassExpectedData.add(new double[]{-0.648352, 0.602339});
        fourclassExpectedData.add(new double[]{-0.395604, -0.730994});
        fourclassExpectedData.add(new double[]{0.208791, 0.0994152});
        fourclassExpectedData.add(new double[]{0.461538, 0.859649});
        fourclassExpectedData.add(new double[]{-0.615385, -0.625731});
        fourclassExpectedData.add(new double[]{0.692308, 0.906433});
        fourclassExpectedData.add(new double[]{-0.626374, -0.169591});
        fourclassExpectedData.add(new double[]{-0.758242, -0.508772});
        fourclassExpectedData.add(new double[]{0.373626, 0.894737});
        fourclassExpectedData.add(new double[]{-0.472527, 0.0760234});
        fourclassExpectedData.add(new double[]{0.318681, 0.0526316});
        fourclassExpectedData.add(new double[]{-0.681319, 0.906433});
        fourclassExpectedData.add(new double[]{0.340659, -0.216374});
        fourclassExpectedData.add(new double[]{-0.604396, 0.719298});
        fourclassExpectedData.add(new double[]{0.626374, 0.871345});
        fourclassExpectedData.add(new double[]{-0.571429, 0.134503});
        fourclassExpectedData.add(new double[]{0.153846, -0.192982});
        fourclassExpectedData.add(new double[]{-0.879121, 0.695906});
        fourclassExpectedData.add(new double[]{-0.648352, -0.169591});
        fourclassExpectedData.add(new double[]{0.230769, 0.251462});
        fourclassExpectedData.add(new double[]{-0.241758, -0.625731});
        fourclassExpectedData.add(new double[]{-0.912088, 0.508772});
        fourclassExpectedData.add(new double[]{0.32967, 0.871345});
        fourclassExpectedData.add(new double[]{-0.835165, 0.450292});
        fourclassExpectedData.add(new double[]{-0.472527, -0.590643});
        fourclassExpectedData.add(new double[]{-0.648352, 0.953216});
        fourclassExpectedData.add(new double[]{-0.802198, -0.660819});
        fourclassExpectedData.add(new double[]{-0.362637, 0.0409357});
        fourclassExpectedData.add(new double[]{-0.956044, 0.649123});
        fourclassExpectedData.add(new double[]{-0.472527, -0.00584795});
        fourclassExpectedData.add(new double[]{-0.197802, -0.0175439});
        fourclassExpectedData.add(new double[]{0.0879121, -0.672515});
        fourclassExpectedData.add(new double[]{0.857143, 0.777778});
        fourclassExpectedData.add(new double[]{-0.835165, -0.497076});
        fourclassExpectedData.add(new double[]{-0.582418, 0.660819});
        fourclassExpectedData.add(new double[]{-0.67033, 0.637427});
        fourclassExpectedData.add(new double[]{-0.351648, -0.0994152});
        fourclassExpectedData.add(new double[]{0.032967, 0.450292});
        fourclassExpectedData.add(new double[]{-0.791209, -0.497076});
        fourclassExpectedData.add(new double[]{-0.582418, 0.216374});
        fourclassExpectedData.add(new double[]{0.285714, 0.216374});
        fourclassExpectedData.add(new double[]{0.153846, -0.0877193});
        fourclassExpectedData.add(new double[]{-0.010989, -0.0643275});
        fourclassExpectedData.add(new double[]{0.285714, 0.111111});
        fourclassExpectedData.add(new double[]{-0.802198, -0.134503});
        fourclassExpectedData.add(new double[]{-0.021978, -0.111111});
        fourclassExpectedData.add(new double[]{0.032967, 0.251462});
        fourclassExpectedData.add(new double[]{-0.835165, -0.403509});
        fourclassExpectedData.add(new double[]{-0.571429, 0.192982});
        fourclassExpectedData.add(new double[]{-0.637363, -0.169591});
        fourclassExpectedData.add(new double[]{-0.307692, -0.0760234});
        fourclassExpectedData.add(new double[]{-0.318681, -0.157895});
        fourclassExpectedData.add(new double[]{0.483516, 0.906433});
        fourclassExpectedData.add(new double[]{-0.428571, 0.122807});
        fourclassExpectedData.add(new double[]{-0.0659341, -0.555556});
        fourclassExpectedData.add(new double[]{-0.549451, 0.22807});
        fourclassExpectedData.add(new double[]{-0.384615, -0.730994});
        fourclassExpectedData.add(new double[]{-0.362637, -0.789474});
        fourclassExpectedData.add(new double[]{-0.461538, 0.660819});
        fourclassExpectedData.add(new double[]{0.758242, 0.824561});
        fourclassExpectedData.add(new double[]{0.813187, 0.894737});
        fourclassExpectedData.add(new double[]{0.175824, -0.964912});
        fourclassExpectedData.add(new double[]{-0.538462, 0.251462});
        fourclassExpectedData.add(new double[]{-0.516484, -0.0643275});
        fourclassExpectedData.add(new double[]{0.0549451, -0.74269});
        fourclassExpectedData.add(new double[]{0.021978, 0.0526316});
        fourclassExpectedData.add(new double[]{-0.0989011, -0.625731});
        fourclassExpectedData.add(new double[]{0.604396, 0.929825});
        fourclassExpectedData.add(new double[]{0.230769, 0.169591});
        fourclassExpectedData.add(new double[]{-0.428571, 0.602339});
        fourclassExpectedData.add(new double[]{-0.043956, -0.918129});
        fourclassExpectedData.add(new double[]{-0.230769, -0.637427});
        fourclassExpectedData.add(new double[]{-0.307692, -0.567251});
        fourclassExpectedData.add(new double[]{-0.626374, -0.0877193});
        fourclassExpectedData.add(new double[]{-0.791209, -0.54386});
        fourclassExpectedData.add(new double[]{-0.340659, 0.0877193});
        fourclassExpectedData.add(new double[]{-0.0989011, 0.0409357});
        fourclassExpectedData.add(new double[]{0.945055, 0.707602});
        fourclassExpectedData.add(new double[]{0.186813, 0.380117});
        fourclassExpectedData.add(new double[]{0.0879121, -1.0});
        fourclassExpectedData.add(new double[]{0.0549451, 0.239766});
        fourclassExpectedData.add(new double[]{-0.373626, -0.625731});
        fourclassExpectedData.add(new double[]{-0.901099, 1.0});
        fourclassExpectedData.add(new double[]{-0.604396, 0.80117});
        fourclassExpectedData.add(new double[]{-0.252747, 0.309942});
        fourclassExpectedData.add(new double[]{-0.274725, -0.789474});
        fourclassExpectedData.add(new double[]{0.208791, 0.28655});
        fourclassExpectedData.add(new double[]{0.472527, -0.0877193});
        fourclassExpectedData.add(new double[]{-0.56044, -0.0409357});
        fourclassExpectedData.add(new double[]{-0.153846, -0.730994});
        fourclassExpectedData.add(new double[]{-0.450549, -0.0409357});
        fourclassExpectedData.add(new double[]{0.285714, 0.309942});
        fourclassExpectedData.add(new double[]{0.21978, -0.777778});
        fourclassExpectedData.add(new double[]{-0.395604, -0.74269});
        fourclassExpectedData.add(new double[]{0.934066, 0.567251});
        fourclassExpectedData.add(new double[]{0.120879, 0.274854});
        fourclassExpectedData.add(new double[]{-0.67033, -0.298246});
        fourclassExpectedData.add(new double[]{-0.659341, -0.0643275});
        fourclassExpectedData.add(new double[]{-0.373626, -0.0643275});
        fourclassExpectedData.add(new double[]{-0.285714, 0.426901});
        fourclassExpectedData.add(new double[]{0.32967, 0.146199});
        fourclassExpectedData.add(new double[]{-0.428571, 0.426901});
        fourclassExpectedData.add(new double[]{0.032967, -0.649123});
        fourclassExpectedData.add(new double[]{-0.626374, -0.497076});
        fourclassExpectedData.add(new double[]{-0.769231, -0.497076});
        fourclassExpectedData.add(new double[]{-0.593407, -0.730994});
        fourclassExpectedData.add(new double[]{-0.582418, -0.824561});
        fourclassExpectedData.add(new double[]{-0.0659341, -0.719298});
        fourclassExpectedData.add(new double[]{-0.604396, 0.134503});
        fourclassExpectedData.add(new double[]{0.373626, 0.0526316});
        fourclassExpectedData.add(new double[]{-0.659341, -0.508772});
        fourclassExpectedData.add(new double[]{-0.692308, -0.80117});
        fourclassExpectedData.add(new double[]{-0.978022, 0.181287});
        fourclassExpectedData.add(new double[]{-0.362637, -0.0175439});
        fourclassExpectedData.add(new double[]{-0.32967, 0.508772});
        fourclassExpectedData.add(new double[]{-0.010989, -0.719298});
        fourclassExpectedData.add(new double[]{0.164835, -0.988304});
        fourclassExpectedData.add(new double[]{-0.56044, 0.00584795});
        fourclassExpectedData.add(new double[]{-0.0989011, -0.777778});
        fourclassExpectedData.add(new double[]{-0.659341, 0.146199});
        fourclassExpectedData.add(new double[]{-0.593407, 0.590643});
        fourclassExpectedData.add(new double[]{-0.56044, 0.649123});
        fourclassExpectedData.add(new double[]{-0.032967, 0.321637});
        fourclassExpectedData.add(new double[]{-0.791209, 0.0526316});
        fourclassExpectedData.add(new double[]{-0.0879121, -0.894737});
        fourclassExpectedData.add(new double[]{0.857143, 0.836257});
        fourclassExpectedData.add(new double[]{-0.549451, 0.859649});
        fourclassExpectedData.add(new double[]{-0.648352, -0.426901});
        fourclassExpectedData.add(new double[]{-0.230769, -0.146199});
        fourclassExpectedData.add(new double[]{-0.593407, 0.263158});
        fourclassExpectedData.add(new double[]{-0.78022, 0.0760234});
        fourclassExpectedData.add(new double[]{-0.032967, 0.415205});
        fourclassExpectedData.add(new double[]{-0.824176, 0.532164});
        fourclassExpectedData.add(new double[]{0.978022, 0.730994});
        fourclassExpectedData.add(new double[]{0.10989, -0.74269});
        fourclassExpectedData.add(new double[]{-0.318681, 0.391813});
        fourclassExpectedData.add(new double[]{-0.835165, -0.216374});
        fourclassExpectedData.add(new double[]{-0.461538, -0.707602});
        fourclassExpectedData.add(new double[]{-0.0659341, -0.0643275});
        fourclassExpectedData.add(new double[]{-0.725275, 0.0643275});
        fourclassExpectedData.add(new double[]{-0.538462, -0.0526316});
        fourclassExpectedData.add(new double[]{-0.483516, -0.812865});
        fourclassExpectedData.add(new double[]{0.901099, 0.684211});
        fourclassExpectedData.add(new double[]{0.274725, 0.28655});
        fourclassExpectedData.add(new double[]{0.89011, 0.871345});
        fourclassExpectedData.add(new double[]{-0.714286, 0.00584795});
        fourclassExpectedData.add(new double[]{0.274725, 0.22807});
        fourclassExpectedData.add(new double[]{-0.703297, 0.847953});
        fourclassExpectedData.add(new double[]{-0.0989011, 0.0526316});
        fourclassExpectedData.add(new double[]{-0.758242, 0.0994152});
        fourclassExpectedData.add(new double[]{-0.615385, 0.22807});
        fourclassExpectedData.add(new double[]{-0.67033, 0.754386});
        fourclassExpectedData.add(new double[]{-0.791209, -0.461988});
        fourclassExpectedData.add(new double[]{-0.120879, -0.0994152});
        fourclassExpectedData.add(new double[]{-0.637363, 0.871345});
        fourclassExpectedData.add(new double[]{0.494505, 0.976608});
        fourclassExpectedData.add(new double[]{-0.538462, 0.730994});
        fourclassExpectedData.add(new double[]{-0.912088, 0.871345});
        fourclassExpectedData.add(new double[]{0.153846, -0.0409357});
        fourclassExpectedData.add(new double[]{0.0659341, -0.146199});
        fourclassExpectedData.add(new double[]{-0.230769, 0.497076});
        fourclassExpectedData.add(new double[]{0.351648, 0.789474});
        fourclassExpectedData.add(new double[]{0.824176, 0.777778});
        fourclassExpectedData.add(new double[]{0.0659341, -0.0526316});
        fourclassExpectedData.add(new double[]{-0.549451, 0.0877193});
        fourclassExpectedData.add(new double[]{-0.912088, 0.157895});
        fourclassExpectedData.add(new double[]{1.0, 0.578947});
        fourclassExpectedData.add(new double[]{0.0989011, 0.380117});
        fourclassExpectedData.add(new double[]{-0.637363, -0.0643275});
        fourclassExpectedData.add(new double[]{0.461538, -0.146199});
        fourclassExpectedData.add(new double[]{0.351648, 0.157895});
        fourclassExpectedData.add(new double[]{-0.43956, 0.461988});
        fourclassExpectedData.add(new double[]{-0.296703, 0.368421});
        fourclassExpectedData.add(new double[]{-0.0659341, 0.380117});
        fourclassExpectedData.add(new double[]{-0.351648, 0.146199});
        fourclassExpectedData.add(new double[]{-0.615385, 0.74269});
        fourclassExpectedData.add(new double[]{-0.923077, 0.695906});
        fourclassExpectedData.add(new double[]{-0.67033, -0.707602});
        fourclassExpectedData.add(new double[]{-0.285714, 0.403509});
        fourclassExpectedData.add(new double[]{-0.637363, 0.766082});
        fourclassExpectedData.add(new double[]{-0.527473, 0.204678});
        fourclassExpectedData.add(new double[]{-0.703297, 0.0409357});
        fourclassExpectedData.add(new double[]{-0.197802, 0.403509});
        fourclassExpectedData.add(new double[]{0.153846, 0.146199});
        fourclassExpectedData.add(new double[]{-0.626374, -0.48538});
        fourclassExpectedData.add(new double[]{0.505495, -0.181287});
        fourclassExpectedData.add(new double[]{-0.703297, 0.766082});
        fourclassExpectedData.add(new double[]{-0.32967, -0.74269});
        fourclassExpectedData.add(new double[]{-0.417582, 0.146199});
        fourclassExpectedData.add(new double[]{0.021978, 0.403509});
        fourclassExpectedData.add(new double[]{-0.593407, 0.578947});
        fourclassExpectedData.add(new double[]{0.175824, -1.0});
        fourclassExpectedData.add(new double[]{0.461538, 0.146199});
        fourclassExpectedData.add(new double[]{-0.362637, 0.625731});
        fourclassExpectedData.add(new double[]{0.21978, 0.894737});
        fourclassExpectedData.add(new double[]{-0.67033, -0.614035});
        fourclassExpectedData.add(new double[]{-0.263736, 0.567251});
        fourclassExpectedData.add(new double[]{0.417582, 0.789474});
        fourclassExpectedData.add(new double[]{0.571429, 0.953216});
        fourclassExpectedData.add(new double[]{-0.032967, -0.169591});
        fourclassExpectedData.add(new double[]{-0.384615, -0.0175439});
        fourclassExpectedData.add(new double[]{0.285714, 0.80117});
        fourclassExpectedData.add(new double[]{-0.626374, -0.672515});
        fourclassExpectedData.add(new double[]{-0.604396, 0.0292398});
        fourclassExpectedData.add(new double[]{-0.197802, 0.0292398});
        fourclassExpectedData.add(new double[]{-0.516484, 0.263158});
        fourclassExpectedData.add(new double[]{-0.637363, 0.00584795});
        fourclassExpectedData.add(new double[]{-0.868132, 0.614035});
        fourclassExpectedData.add(new double[]{-0.417582, 0.111111});
        fourclassExpectedData.add(new double[]{-0.461538, -0.590643});
        fourclassExpectedData.add(new double[]{0.10989, -0.146199});
        fourclassExpectedData.add(new double[]{-0.582418, 0.649123});
        fourclassExpectedData.add(new double[]{0.142857, -0.730994});
        fourclassExpectedData.add(new double[]{-0.406593, 0.590643});
        fourclassExpectedData.add(new double[]{0.10989, -0.789474});
        fourclassExpectedData.add(new double[]{-0.637363, 0.672515});
        fourclassExpectedData.add(new double[]{-0.56044, 0.74269});
        fourclassExpectedData.add(new double[]{-0.648352, 0.824561});
        fourclassExpectedData.add(new double[]{0.923077, 0.707602});
        fourclassExpectedData.add(new double[]{-0.307692, 0.0760234});
        fourclassExpectedData.add(new double[]{0.351648, -0.146199});
        fourclassExpectedData.add(new double[]{-0.538462, -0.695906});
        fourclassExpectedData.add(new double[]{-0.538462, 0.614035});
        fourclassExpectedData.add(new double[]{-0.472527, -0.0526316});
        fourclassExpectedData.add(new double[]{-0.89011, 0.263158});
        fourclassExpectedData.add(new double[]{-0.483516, 0.239766});
        fourclassExpectedData.add(new double[]{-0.296703, -0.0292398});
        fourclassExpectedData.add(new double[]{-0.802198, 0.602339});
        fourclassExpectedData.add(new double[]{-0.131868, -0.0409357});
        fourclassExpectedData.add(new double[]{-0.945055, 0.54386});
        fourclassExpectedData.add(new double[]{0.296703, 0.239766});
        fourclassExpectedData.add(new double[]{-0.318681, 0.508772});
        fourclassExpectedData.add(new double[]{-0.252747, -0.614035});
        fourclassExpectedData.add(new double[]{-0.131868, 0.0409357});
        fourclassExpectedData.add(new double[]{-0.78022, -0.730994});
        fourclassExpectedData.add(new double[]{-0.417582, 0.590643});
        fourclassExpectedData.add(new double[]{0.351648, 0.80117});
        fourclassExpectedData.add(new double[]{0.615385, 0.871345});
        fourclassExpectedData.add(new double[]{-0.956044, 0.239766});
        fourclassExpectedData.add(new double[]{0.153846, 0.309942});
        fourclassExpectedData.add(new double[]{-0.89011, 0.497076});
        fourclassExpectedData.add(new double[]{0.362637, 0.0175439});
        fourclassExpectedData.add(new double[]{-0.912088, 0.309942});
        fourclassExpectedData.add(new double[]{-0.010989, 0.0292398});
        fourclassExpectedData.add(new double[]{-0.0549451, -0.0643275});
        fourclassExpectedData.add(new double[]{-0.417582, -0.602339});
        fourclassExpectedData.add(new double[]{-0.428571, -0.672515});
        fourclassExpectedData.add(new double[]{-0.186813, -0.719298});
        fourclassExpectedData.add(new double[]{0.538462, 0.906433});
        fourclassExpectedData.add(new double[]{0.901099, 0.602339});
        fourclassExpectedData.add(new double[]{0.0549451, -0.660819});
        fourclassExpectedData.add(new double[]{0.868132, 0.672515});
        fourclassExpectedData.add(new double[]{-0.021978, 0.356725});
        fourclassExpectedData.add(new double[]{-0.395604, 0.216374});
        fourclassExpectedData.add(new double[]{-0.0879121, 0.426901});
        fourclassExpectedData.add(new double[]{0.32967, 0.28655});
        fourclassExpectedData.add(new double[]{0.021978, 0.0409357});
        fourclassExpectedData.add(new double[]{-0.131868, -0.812865});
        fourclassExpectedData.add(new double[]{0.153846, 0.169591});
        fourclassExpectedData.add(new double[]{-0.197802, -0.00584795});
        fourclassExpectedData.add(new double[]{0.208791, 0.157895});
        fourclassExpectedData.add(new double[]{0.241758, 0.894737});
        fourclassExpectedData.add(new double[]{-0.296703, -0.777778});
        fourclassExpectedData.add(new double[]{-0.0989011, -0.660819});
        fourclassExpectedData.add(new double[]{-0.604396, -0.567251});
        fourclassExpectedData.add(new double[]{-0.010989, 0.0643275});
        fourclassExpectedData.add(new double[]{-0.824176, -0.391813});
        fourclassExpectedData.add(new double[]{-0.241758, 0.0175439});
        fourclassExpectedData.add(new double[]{-0.021978, -0.146199});
        fourclassExpectedData.add(new double[]{0.197802, 0.263158});
        fourclassExpectedData.add(new double[]{-0.307692, -0.111111});
        fourclassExpectedData.add(new double[]{-0.681319, -0.625731});
        fourclassExpectedData.add(new double[]{-0.604396, 0.251462});
        fourclassExpectedData.add(new double[]{-0.978022, 0.438596});
        fourclassExpectedData.add(new double[]{-0.043956, -0.157895});
        fourclassExpectedData.add(new double[]{-0.494505, 0.649123});
        fourclassExpectedData.add(new double[]{0.043956, -0.0994152});
        fourclassExpectedData.add(new double[]{-0.835165, -0.0994152});
        fourclassExpectedData.add(new double[]{-0.0549451, -0.192982});
        fourclassExpectedData.add(new double[]{0.571429, 0.894737});
        fourclassExpectedData.add(new double[]{-0.131868, -0.847953});
        fourclassExpectedData.add(new double[]{-0.791209, -0.274854});
        fourclassExpectedData.add(new double[]{-0.791209, -0.695906});
        fourclassExpectedData.add(new double[]{-0.527473, 0.0409357});
        fourclassExpectedData.add(new double[]{-0.0769231, 0.333333});
        fourclassExpectedData.add(new double[]{0.89011, 0.812865});
        fourclassExpectedData.add(new double[]{0.604396, 0.953216});
        fourclassExpectedData.add(new double[]{-0.538462, 0.146199});
        fourclassExpectedData.add(new double[]{-0.0989011, 0.298246});
        fourclassExpectedData.add(new double[]{0.0989011, 0.192982});
        fourclassExpectedData.add(new double[]{-0.604396, 0.812865});
        fourclassExpectedData.add(new double[]{0.67033, 0.906433});
        fourclassExpectedData.add(new double[]{-0.857143, 0.567251});
        fourclassExpectedData.add(new double[]{0.725275, 0.906433});
        fourclassExpectedData.add(new double[]{-0.626374, 0.192982});
        fourclassExpectedData.add(new double[]{-0.021978, -0.0760234});
        fourclassExpectedData.add(new double[]{0.340659, 0.812865});
        fourclassExpectedData.add(new double[]{-0.318681, 0.450292});
        fourclassExpectedData.add(new double[]{0.450549, 0.906433});
        fourclassExpectedData.add(new double[]{-0.43956, -0.754386});
        fourclassExpectedData.add(new double[]{-0.923077, 0.309942});
        fourclassExpectedData.add(new double[]{-0.384615, 0.0760234});
        fourclassExpectedData.add(new double[]{-0.10989, -0.0526316});
        fourclassExpectedData.add(new double[]{0.120879, 0.216374});
        fourclassExpectedData.add(new double[]{-0.912088, 0.497076});
        fourclassExpectedData.add(new double[]{-0.373626, -0.578947});
        fourclassExpectedData.add(new double[]{-0.406593, -0.555556});
        fourclassExpectedData.add(new double[]{0.56044, 0.976608});
        fourclassExpectedData.add(new double[]{-0.736264, -0.684211});
        fourclassExpectedData.add(new double[]{0.285714, 0.906433});
        fourclassExpectedData.add(new double[]{0.120879, 0.239766});
        fourclassExpectedData.add(new double[]{-0.131868, -0.730994});
        fourclassExpectedData.add(new double[]{0.494505, 0.122807});
        fourclassExpectedData.add(new double[]{-0.131868, 0.00584795});
        fourclassExpectedData.add(new double[]{-0.252747, -0.0175439});
        fourclassExpectedData.add(new double[]{-0.604396, 0.918129});
        fourclassExpectedData.add(new double[]{-0.912088, 0.649123});
        fourclassExpectedData.add(new double[]{-0.164835, -0.824561});
        fourclassExpectedData.add(new double[]{-0.879121, 0.976608});
        fourclassExpectedData.add(new double[]{0.417582, -0.169591});
        fourclassExpectedData.add(new double[]{-0.483516, 0.0526316});
        fourclassExpectedData.add(new double[]{0.230769, 0.871345});
        fourclassExpectedData.add(new double[]{0.0769231, 0.251462});
        fourclassExpectedData.add(new double[]{-0.351648, 0.403509});
        fourclassExpectedData.add(new double[]{0.32967, 0.918129});
        fourclassExpectedData.add(new double[]{-0.835165, 0.859649});
        fourclassExpectedData.add(new double[]{-0.824176, 0.918129});
        fourclassExpectedData.add(new double[]{-0.604396, -0.146199});
        fourclassExpectedData.add(new double[]{-0.824176, 0.859649});
        fourclassExpectedData.add(new double[]{-0.472527, -0.824561});
        fourclassExpectedData.add(new double[]{-0.747253, 0.0292398});
        fourclassExpectedData.add(new double[]{-0.142857, -0.555556});
        fourclassExpectedData.add(new double[]{0.483516, 0.122807});
        fourclassExpectedData.add(new double[]{-0.450549, 0.450292});
        fourclassExpectedData.add(new double[]{-0.142857, -0.754386});
        fourclassExpectedData.add(new double[]{-0.604396, 0.0760234});
        fourclassExpectedData.add(new double[]{0.714286, 0.918129});
        fourclassExpectedData.add(new double[]{-0.626374, 0.0994152});
        fourclassExpectedData.add(new double[]{-0.626374, 0.0877193});
        fourclassExpectedData.add(new double[]{-0.571429, 0.263158});
        fourclassExpectedData.add(new double[]{0.142857, -0.812865});
        fourclassExpectedData.add(new double[]{-0.648352, -0.438596});
        fourclassExpectedData.add(new double[]{-0.021978, 0.345029});
        fourclassExpectedData.add(new double[]{-0.835165, 0.438596});
        fourclassExpectedData.add(new double[]{-0.197802, -0.54386});
        fourclassExpectedData.add(new double[]{-0.593407, 0.602339});
        fourclassExpectedData.add(new double[]{-0.824176, 0.508772});
        fourclassExpectedData.add(new double[]{0.010989, 0.403509});
        fourclassExpectedData.add(new double[]{-0.78022, 1.0});
        fourclassExpectedData.add(new double[]{-0.857143, 0.54386});
        fourclassExpectedData.add(new double[]{-0.626374, -0.415205});
        fourclassExpectedData.add(new double[]{-0.483516, 0.450292});
        fourclassExpectedData.add(new double[]{0.78022, 0.754386});
        fourclassExpectedData.add(new double[]{1.0, 0.649123});
        fourclassExpectedData.add(new double[]{-0.010989, 0.403509});
        fourclassExpectedData.add(new double[]{0.230769, -0.871345});
        fourclassExpectedData.add(new double[]{-0.615385, 0.847953});
        fourclassExpectedData.add(new double[]{0.274725, 0.789474});
        fourclassExpectedData.add(new double[]{-0.043956, -0.836257});
        fourclassExpectedData.add(new double[]{-0.714286, -0.719298});
        fourclassExpectedData.add(new double[]{-0.505495, 0.0877193});
        fourclassExpectedData.add(new double[]{0.428571, 0.953216});
        fourclassExpectedData.add(new double[]{-0.912088, 0.28655});
        fourclassExpectedData.add(new double[]{0.142857, 0.00584795});
        return fourclassExpectedData;
    }

    public List<double[]> createAnomalyData() {
        List<double[]> fourclassAnomalyData = new ArrayList<>();
        fourclassAnomalyData.add(new double[]{0.692308, -0.824561});
        fourclassAnomalyData.add(new double[]{0.769231, -0.321637});
        fourclassAnomalyData.add(new double[]{-0.21978, -0.263158});
        fourclassAnomalyData.add(new double[]{0.318681, 0.54386});
        fourclassAnomalyData.add(new double[]{0.0989011, -0.309942});
        fourclassAnomalyData.add(new double[]{0.912088, -0.730994});
        fourclassAnomalyData.add(new double[]{0.912088, 0.181287});
        fourclassAnomalyData.add(new double[]{0.615385, -0.812865});
        fourclassAnomalyData.add(new double[]{0.912088, 0.0760234});
        fourclassAnomalyData.add(new double[]{0.78022, -0.707602});
        fourclassAnomalyData.add(new double[]{0.197802, 0.567251});
        fourclassAnomalyData.add(new double[]{0.912088, -0.333333});
        fourclassAnomalyData.add(new double[]{0.769231, -0.637427});
        fourclassAnomalyData.add(new double[]{0.78022, -0.918129});
        fourclassAnomalyData.add(new double[]{0.296703, 0.497076});
        fourclassAnomalyData.add(new double[]{0.527473, -0.684211});
        fourclassAnomalyData.add(new double[]{0.626374, 0.497076});
        fourclassAnomalyData.add(new double[]{0.0879121, -0.368421});
        fourclassAnomalyData.add(new double[]{-0.296703, -0.298246});
        fourclassAnomalyData.add(new double[]{0.758242, 0.345029});
        fourclassAnomalyData.add(new double[]{0.912088, 0.0409357});
        fourclassAnomalyData.add(new double[]{0.417582, 0.602339});
        fourclassAnomalyData.add(new double[]{0.362637, -0.649123});
        fourclassAnomalyData.add(new double[]{0.89011, -0.181287});
        fourclassAnomalyData.add(new double[]{0.901099, -0.74269});
        fourclassAnomalyData.add(new double[]{0.604396, -0.929825});
        fourclassAnomalyData.add(new double[]{0.494505, -0.707602});
        fourclassAnomalyData.add(new double[]{0.758242, 0.204678});
        fourclassAnomalyData.add(new double[]{0.769231, -0.777778});
        fourclassAnomalyData.add(new double[]{0.384615, 0.497076});
        fourclassAnomalyData.add(new double[]{0.879121, -0.578947});
        fourclassAnomalyData.add(new double[]{0.318681, 0.672515});
        fourclassAnomalyData.add(new double[]{0.549451, 0.590643});
        fourclassAnomalyData.add(new double[]{0.835165, -0.368421});
        fourclassAnomalyData.add(new double[]{0.285714, -0.48538});
        fourclassAnomalyData.add(new double[]{0.758242, -0.321637});
        fourclassAnomalyData.add(new double[]{0.791209, -0.181287});
        fourclassAnomalyData.add(new double[]{0.813187, -0.239766});
        fourclassAnomalyData.add(new double[]{0.208791, -0.590643});
        fourclassAnomalyData.add(new double[]{0.824176, 0.368421});
        fourclassAnomalyData.add(new double[]{0.0549451, -0.497076});
        fourclassAnomalyData.add(new double[]{0.252747, 0.567251});
        fourclassAnomalyData.add(new double[]{0.197802, 0.637427});
        fourclassAnomalyData.add(new double[]{0.230769, 0.602339});
        fourclassAnomalyData.add(new double[]{-0.0989011, 0.649123});
        fourclassAnomalyData.add(new double[]{0.846154, -0.906433});
        fourclassAnomalyData.add(new double[]{0.362637, 0.637427});
        fourclassAnomalyData.add(new double[]{0.197802, -0.567251});
        fourclassAnomalyData.add(new double[]{-0.142857, 0.660819});
        fourclassAnomalyData.add(new double[]{0.67033, 0.309942});
        fourclassAnomalyData.add(new double[]{0.428571, -0.660819});
        fourclassAnomalyData.add(new double[]{-0.0989011, 0.754386});
        fourclassAnomalyData.add(new double[]{0.384615, 0.473684});
        fourclassAnomalyData.add(new double[]{0.10989, 0.520468});
        fourclassAnomalyData.add(new double[]{0.318681, 0.614035});
        fourclassAnomalyData.add(new double[]{0.857143, 0.368421});
        fourclassAnomalyData.add(new double[]{0.791209, -0.707602});
        fourclassAnomalyData.add(new double[]{0.197802, 0.649123});
        fourclassAnomalyData.add(new double[]{-0.351648, -0.391813});
        fourclassAnomalyData.add(new double[]{0.263736, 0.532164});
        fourclassAnomalyData.add(new double[]{0.758242, 0.169591});
        fourclassAnomalyData.add(new double[]{0.505495, 0.461988});
        fourclassAnomalyData.add(new double[]{0.197802, -0.555556});
        fourclassAnomalyData.add(new double[]{0.923077, -0.80117});
        fourclassAnomalyData.add(new double[]{-0.043956, 0.707602});
        fourclassAnomalyData.add(new double[]{0.56044, -0.730994});
        fourclassAnomalyData.add(new double[]{0.241758, -0.567251});
        fourclassAnomalyData.add(new double[]{0.912088, -0.356725});
        fourclassAnomalyData.add(new double[]{-0.175824, 0.672515});
        fourclassAnomalyData.add(new double[]{0.758242, -0.169591});
        fourclassAnomalyData.add(new double[]{0.648352, -0.929825});
        fourclassAnomalyData.add(new double[]{0.67033, -1.0});
        fourclassAnomalyData.add(new double[]{0.0659341, 0.695906});
        fourclassAnomalyData.add(new double[]{0.483516, -0.789474});
        fourclassAnomalyData.add(new double[]{0.043956, -0.28655});
        fourclassAnomalyData.add(new double[]{0.846154, -0.988304});
        fourclassAnomalyData.add(new double[]{0.32967, -0.719298});
        fourclassAnomalyData.add(new double[]{-0.10989, -0.298246});
        fourclassAnomalyData.add(new double[]{-0.175824, 0.812865});
        fourclassAnomalyData.add(new double[]{0.769231, 0.0175439});
        fourclassAnomalyData.add(new double[]{0.450549, -0.871345});
        fourclassAnomalyData.add(new double[]{0.0879121, 0.660819});
        fourclassAnomalyData.add(new double[]{0.32967, 0.567251});
        fourclassAnomalyData.add(new double[]{0.593407, 0.461988});
        fourclassAnomalyData.add(new double[]{0.901099, -0.824561});
        fourclassAnomalyData.add(new double[]{0.21978, -0.461988});
        fourclassAnomalyData.add(new double[]{0.89011, -0.415205});
        fourclassAnomalyData.add(new double[]{0.0659341, -0.426901});
        fourclassAnomalyData.add(new double[]{0.56044, 0.415205});
        fourclassAnomalyData.add(new double[]{0.582418, 0.426901});
        fourclassAnomalyData.add(new double[]{-0.010989, 0.602339});
        fourclassAnomalyData.add(new double[]{-0.043956, -0.391813});
        fourclassAnomalyData.add(new double[]{0.879121, -0.0175439});
        fourclassAnomalyData.add(new double[]{0.186813, -0.520468});
        fourclassAnomalyData.add(new double[]{0.725275, 0.345029});
        fourclassAnomalyData.add(new double[]{0.142857, -0.555556});
        fourclassAnomalyData.add(new double[]{0.747253, 0.263158});
        fourclassAnomalyData.add(new double[]{0.56044, -0.684211});
        fourclassAnomalyData.add(new double[]{0.450549, 0.48538});
        fourclassAnomalyData.add(new double[]{0.340659, 0.625731});
        fourclassAnomalyData.add(new double[]{0.10989, -0.368421});
        fourclassAnomalyData.add(new double[]{0.824176, -0.298246});
        fourclassAnomalyData.add(new double[]{0.175824, -0.368421});
        fourclassAnomalyData.add(new double[]{-0.0659341, 0.614035});
        fourclassAnomalyData.add(new double[]{0.318681, 0.497076});
        fourclassAnomalyData.add(new double[]{0.318681, 0.532164});
        fourclassAnomalyData.add(new double[]{0.307692, -0.614035});
        fourclassAnomalyData.add(new double[]{0.516484, 0.508772});
        fourclassAnomalyData.add(new double[]{0.747253, -0.988304});
        fourclassAnomalyData.add(new double[]{0.505495, -0.859649});
        fourclassAnomalyData.add(new double[]{0.912088, -0.426901});
        fourclassAnomalyData.add(new double[]{0.846154, -0.824561});
        fourclassAnomalyData.add(new double[]{0.538462, 0.391813});
        fourclassAnomalyData.add(new double[]{0.241758, 0.578947});
        fourclassAnomalyData.add(new double[]{0.835165, -0.532164});
        fourclassAnomalyData.add(new double[]{0.021978, -0.391813});
        fourclassAnomalyData.add(new double[]{0.307692, 0.625731});
        fourclassAnomalyData.add(new double[]{-0.252747, -0.309942});
        fourclassAnomalyData.add(new double[]{0.857143, -0.508772});
        fourclassAnomalyData.add(new double[]{0.835165, -0.80117});
        fourclassAnomalyData.add(new double[]{-0.274725, -0.333333});
        fourclassAnomalyData.add(new double[]{-0.274725, -0.251462});
        fourclassAnomalyData.add(new double[]{-0.032967, 0.532164});
        fourclassAnomalyData.add(new double[]{0.0769231, -0.298246});
        fourclassAnomalyData.add(new double[]{0.725275, 0.415205});
        fourclassAnomalyData.add(new double[]{0.835165, 0.321637});
        fourclassAnomalyData.add(new double[]{0.78022, -0.660819});
        fourclassAnomalyData.add(new double[]{0.362637, 0.532164});
        fourclassAnomalyData.add(new double[]{0.307692, -0.520468});
        fourclassAnomalyData.add(new double[]{0.868132, 0.169591});
        fourclassAnomalyData.add(new double[]{0.824176, -0.122807});
        fourclassAnomalyData.add(new double[]{0.021978, -0.380117});
        fourclassAnomalyData.add(new double[]{0.758242, -0.146199});
        fourclassAnomalyData.add(new double[]{0.494505, 0.637427});
        fourclassAnomalyData.add(new double[]{0.494505, -0.74269});
        fourclassAnomalyData.add(new double[]{0.681319, 0.321637});
        fourclassAnomalyData.add(new double[]{0.428571, 0.649123});
        fourclassAnomalyData.add(new double[]{0.153846, 0.672515});
        fourclassAnomalyData.add(new double[]{-0.032967, -0.321637});
        fourclassAnomalyData.add(new double[]{-0.120879, -0.356725});
        fourclassAnomalyData.add(new double[]{0.692308, 0.333333});
        fourclassAnomalyData.add(new double[]{0.153846, -0.321637});
        fourclassAnomalyData.add(new double[]{0.857143, -0.754386});
        fourclassAnomalyData.add(new double[]{0.769231, -0.555556});
        fourclassAnomalyData.add(new double[]{0.703297, -0.789474});
        fourclassAnomalyData.add(new double[]{0.857143, -0.812865});
        fourclassAnomalyData.add(new double[]{0.0879121, -0.497076});
        fourclassAnomalyData.add(new double[]{0.791209, -0.204678});
        fourclassAnomalyData.add(new double[]{0.395604, -0.578947});
        fourclassAnomalyData.add(new double[]{0.318681, -0.660819});
        fourclassAnomalyData.add(new double[]{-0.395604, -0.345029});
        fourclassAnomalyData.add(new double[]{0.857143, -0.169591});
        fourclassAnomalyData.add(new double[]{0.835165, -0.94152});
        fourclassAnomalyData.add(new double[]{0.857143, -0.684211});
        fourclassAnomalyData.add(new double[]{0.747253, 0.169591});
        fourclassAnomalyData.add(new double[]{0.703297, -0.80117});
        fourclassAnomalyData.add(new double[]{0.571429, -0.836257});
        fourclassAnomalyData.add(new double[]{0.813187, -0.976608});
        fourclassAnomalyData.add(new double[]{0.846154, 0.333333});
        fourclassAnomalyData.add(new double[]{0.021978, -0.438596});
        fourclassAnomalyData.add(new double[]{0.395604, -0.48538});
        fourclassAnomalyData.add(new double[]{0.791209, 0.0994152});
        fourclassAnomalyData.add(new double[]{0.857143, -0.22807});
        fourclassAnomalyData.add(new double[]{-0.153846, 0.719298});
        fourclassAnomalyData.add(new double[]{-0.10989, 0.625731});
        fourclassAnomalyData.add(new double[]{0.252747, -0.48538});
        fourclassAnomalyData.add(new double[]{-0.428571, -0.309942});
        fourclassAnomalyData.add(new double[]{-0.0659341, -0.450292});
        fourclassAnomalyData.add(new double[]{0.043956, 0.614035});
        fourclassAnomalyData.add(new double[]{0.802198, 0.239766});
        fourclassAnomalyData.add(new double[]{-0.120879, 0.695906});
        fourclassAnomalyData.add(new double[]{0.450549, -0.567251});
        fourclassAnomalyData.add(new double[]{0.120879, 0.555556});
        fourclassAnomalyData.add(new double[]{0.725275, 0.380117});
        fourclassAnomalyData.add(new double[]{0.791209, -0.847953});
        fourclassAnomalyData.add(new double[]{0.032967, -0.438596});
        fourclassAnomalyData.add(new double[]{-0.164835, 0.847953});
        fourclassAnomalyData.add(new double[]{0.89011, 0.251462});
        fourclassAnomalyData.add(new double[]{-0.142857, -0.380117});
        fourclassAnomalyData.add(new double[]{0.912088, -0.567251});
        fourclassAnomalyData.add(new double[]{0.538462, -0.637427});
        fourclassAnomalyData.add(new double[]{0.494505, -0.719298});
        fourclassAnomalyData.add(new double[]{0.857143, -0.298246});
        fourclassAnomalyData.add(new double[]{0.285714, -0.672515});
        fourclassAnomalyData.add(new double[]{-0.230769, -0.274854});
        fourclassAnomalyData.add(new double[]{0.802198, -0.426901});
        fourclassAnomalyData.add(new double[]{0.241758, 0.508772});
        fourclassAnomalyData.add(new double[]{0.813187, -0.836257});
        fourclassAnomalyData.add(new double[]{0.67033, 0.473684});
        fourclassAnomalyData.add(new double[]{0.846154, 0.251462});
        fourclassAnomalyData.add(new double[]{0.846154, -0.0760234});
        fourclassAnomalyData.add(new double[]{0.516484, -0.637427});
        fourclassAnomalyData.add(new double[]{0.791209, 0.450292});
        fourclassAnomalyData.add(new double[]{0.197802, -0.461988});
        fourclassAnomalyData.add(new double[]{0.406593, -0.508772});
        fourclassAnomalyData.add(new double[]{0.274725, 0.520468});
        fourclassAnomalyData.add(new double[]{-0.0989011, 0.847953});
        fourclassAnomalyData.add(new double[]{0.494505, -0.906433});
        fourclassAnomalyData.add(new double[]{0.549451, 0.461988});
        fourclassAnomalyData.add(new double[]{0.494505, 0.602339});
        fourclassAnomalyData.add(new double[]{0.912088, -0.0994152});
        fourclassAnomalyData.add(new double[]{-0.252747, -0.450292});
        fourclassAnomalyData.add(new double[]{-0.010989, 0.766082});
        fourclassAnomalyData.add(new double[]{0.846154, -0.953216});
        fourclassAnomalyData.add(new double[]{0.142857, -0.345029});
        fourclassAnomalyData.add(new double[]{-0.252747, -0.333333});
        fourclassAnomalyData.add(new double[]{0.241758, -0.415205});
        fourclassAnomalyData.add(new double[]{0.351648, -0.461988});
        fourclassAnomalyData.add(new double[]{0.0769231, -0.274854});
        fourclassAnomalyData.add(new double[]{-0.010989, 0.578947});
        fourclassAnomalyData.add(new double[]{0.571429, -0.74269});
        fourclassAnomalyData.add(new double[]{0.901099, -0.0643275});
        fourclassAnomalyData.add(new double[]{0.230769, -0.590643});
        fourclassAnomalyData.add(new double[]{0.923077, -0.695906});
        fourclassAnomalyData.add(new double[]{-0.142857, 0.789474});
        fourclassAnomalyData.add(new double[]{0.791209, -0.988304});
        fourclassAnomalyData.add(new double[]{0.43956, 0.567251});
        fourclassAnomalyData.add(new double[]{0.725275, 0.204678});
        fourclassAnomalyData.add(new double[]{-0.142857, 0.625731});
        fourclassAnomalyData.add(new double[]{0.868132, 0.309942});
        fourclassAnomalyData.add(new double[]{0.428571, -0.614035});
        fourclassAnomalyData.add(new double[]{0.626374, 0.578947});
        fourclassAnomalyData.add(new double[]{0.483516, -0.894737});
        fourclassAnomalyData.add(new double[]{0.857143, -0.134503});
        fourclassAnomalyData.add(new double[]{0.274725, -0.461988});
        fourclassAnomalyData.add(new double[]{-0.472527, -0.263158});
        fourclassAnomalyData.add(new double[]{0.0769231, 0.707602});
        fourclassAnomalyData.add(new double[]{0.879121, -0.263158});
        fourclassAnomalyData.add(new double[]{0.769231, -1.0});
        fourclassAnomalyData.add(new double[]{-0.197802, -0.298246});
        fourclassAnomalyData.add(new double[]{-0.263736, -0.274854});
        fourclassAnomalyData.add(new double[]{0.89011, -0.0409357});
        fourclassAnomalyData.add(new double[]{-0.307692, -0.356725});
        fourclassAnomalyData.add(new double[]{0.714286, 0.263158});
        fourclassAnomalyData.add(new double[]{0.637363, 0.391813});
        fourclassAnomalyData.add(new double[]{0.351648, 0.508772});
        fourclassAnomalyData.add(new double[]{-0.10989, -0.368421});
        fourclassAnomalyData.add(new double[]{0.758242, -0.672515});
        fourclassAnomalyData.add(new double[]{-0.406593, -0.356725});
        fourclassAnomalyData.add(new double[]{-0.164835, -0.28655});
        fourclassAnomalyData.add(new double[]{0.835165, -0.590643});
        fourclassAnomalyData.add(new double[]{0.516484, -0.894737});
        fourclassAnomalyData.add(new double[]{0.89011, 0.146199});
        fourclassAnomalyData.add(new double[]{-0.032967, 0.707602});
        fourclassAnomalyData.add(new double[]{0.461538, -0.871345});
        fourclassAnomalyData.add(new double[]{0.417582, -0.707602});
        fourclassAnomalyData.add(new double[]{0.142857, 0.684211});
        fourclassAnomalyData.add(new double[]{0.637363, -0.929825});
        fourclassAnomalyData.add(new double[]{0.021978, -0.426901});
        fourclassAnomalyData.add(new double[]{0.78022, -0.74269});
        fourclassAnomalyData.add(new double[]{0.791209, -0.929825});
        fourclassAnomalyData.add(new double[]{0.0769231, -0.403509});
        fourclassAnomalyData.add(new double[]{0.67033, -0.847953});
        fourclassAnomalyData.add(new double[]{0.241758, 0.555556});
        fourclassAnomalyData.add(new double[]{0.010989, 0.637427});
        fourclassAnomalyData.add(new double[]{0.032967, -0.28655});
        fourclassAnomalyData.add(new double[]{0.0989011, 0.508772});
        fourclassAnomalyData.add(new double[]{-0.230769, -0.426901});
        fourclassAnomalyData.add(new double[]{0.846154, -0.321637});
        fourclassAnomalyData.add(new double[]{0.626374, -0.953216});
        fourclassAnomalyData.add(new double[]{0.89011, -0.0760234});
        fourclassAnomalyData.add(new double[]{0.846154, -0.22807});
        fourclassAnomalyData.add(new double[]{-0.010989, -0.356725});
        fourclassAnomalyData.add(new double[]{-0.043956, 0.637427});
        fourclassAnomalyData.add(new double[]{-0.043956, -0.368421});
        fourclassAnomalyData.add(new double[]{-0.417582, -0.356725});
        fourclassAnomalyData.add(new double[]{0.912088, -0.192982});
        fourclassAnomalyData.add(new double[]{-0.395604, -0.380117});
        fourclassAnomalyData.add(new double[]{0.175824, -0.356725});
        fourclassAnomalyData.add(new double[]{0.725275, -0.847953});
        fourclassAnomalyData.add(new double[]{0.857143, -0.28655});
        fourclassAnomalyData.add(new double[]{0.318681, -0.54386});
        fourclassAnomalyData.add(new double[]{0.813187, -0.637427});
        fourclassAnomalyData.add(new double[]{-0.0989011, 0.859649});
        fourclassAnomalyData.add(new double[]{-0.274725, -0.321637});
        fourclassAnomalyData.add(new double[]{0.791209, -0.380117});
        fourclassAnomalyData.add(new double[]{0.769231, 0.461988});
        fourclassAnomalyData.add(new double[]{0.021978, -0.274854});
        fourclassAnomalyData.add(new double[]{0.571429, -0.777778});
        fourclassAnomalyData.add(new double[]{0.307692, -0.578947});
        fourclassAnomalyData.add(new double[]{0.461538, 0.450292});
        fourclassAnomalyData.add(new double[]{-0.153846, -0.426901});
        fourclassAnomalyData.add(new double[]{0.912088, -0.0175439});
        fourclassAnomalyData.add(new double[]{0.0549451, -0.368421});
        fourclassAnomalyData.add(new double[]{0.78022, -0.695906});
        fourclassAnomalyData.add(new double[]{0.89011, -0.380117});
        fourclassAnomalyData.add(new double[]{-0.43956, -0.391813});
        fourclassAnomalyData.add(new double[]{0.241758, 0.532164});
        fourclassAnomalyData.add(new double[]{0.252747, -0.497076});
        fourclassAnomalyData.add(new double[]{-0.0989011, -0.461988});
        fourclassAnomalyData.add(new double[]{0.351648, -0.637427});
        fourclassAnomalyData.add(new double[]{0.538462, -0.859649});
        fourclassAnomalyData.add(new double[]{0.494505, -0.625731});
        fourclassAnomalyData.add(new double[]{0.78022, -0.28655});
        fourclassAnomalyData.add(new double[]{-0.307692, -0.333333});
        fourclassAnomalyData.add(new double[]{-0.241758, -0.28655});
        fourclassAnomalyData.add(new double[]{0.472527, -0.578947});
        fourclassAnomalyData.add(new double[]{-0.208791, -0.415205});
        fourclassAnomalyData.add(new double[]{-0.0769231, 0.789474});
        fourclassAnomalyData.add(new double[]{-0.0549451, 0.80117});
        fourclassAnomalyData.add(new double[]{0.868132, -0.637427});
        fourclassAnomalyData.add(new double[]{0.868132, 0.251462});
        fourclassAnomalyData.add(new double[]{-0.0549451, 0.567251});
        fourclassAnomalyData.add(new double[]{0.043956, 0.730994});
        fourclassAnomalyData.add(new double[]{0.835165, -0.0760234});
        return fourclassAnomalyData;
    }

}