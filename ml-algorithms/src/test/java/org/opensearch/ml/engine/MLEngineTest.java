/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine;

import com.amazon.randomcutforest.RandomCutForest;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataset.DataFrameInputDataset;
import org.opensearch.ml.common.dataset.MLInputDataset;
import org.opensearch.ml.common.parameter.Input;
import org.opensearch.ml.common.parameter.KMeansParams;
import org.opensearch.ml.common.parameter.LinearRegressionParams;
import org.opensearch.ml.common.parameter.FunctionName;
import org.opensearch.ml.common.parameter.MLAlgoParams;
import org.opensearch.ml.common.parameter.MLInput;
import org.opensearch.ml.common.parameter.Model;
import org.opensearch.ml.common.parameter.MLPredictionOutput;


import java.util.ArrayList;
import java.util.List;

import static org.opensearch.ml.engine.helper.KMeansHelper.constructKMeansDataFrame;
import static org.opensearch.ml.engine.helper.LinearRegressionHelper.constructLinearRegressionPredictionDataFrame;
import static org.opensearch.ml.engine.helper.LinearRegressionHelper.constructLinearRegressionTrainDataFrame;

public class MLEngineTest {
    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();

    @Test
    public void predictKMeans() {
        Model model = trainKMeansModel();
        DataFrame predictionDataFrame = constructKMeansDataFrame(10);
        MLInputDataset inputDataset = DataFrameInputDataset.builder().dataFrame(predictionDataFrame).build();
        Input mlInput = MLInput.builder().algorithm(FunctionName.KMEANS).inputDataset(inputDataset).build();
        MLPredictionOutput output = (MLPredictionOutput)MLEngine.predict(mlInput, model);
        DataFrame predictions = output.getPredictionResult();
        Assert.assertEquals(10, predictions.size());
        predictions.forEach(row -> Assert.assertTrue(row.getValue(0).intValue() == 0 || row.getValue(0).intValue() == 1));
    }

    @Test
    public void predictLinearRegression() {
        Model model = trainLinearRegressionModel();
        DataFrame predictionDataFrame = constructLinearRegressionPredictionDataFrame();
        MLInputDataset inputDataset = DataFrameInputDataset.builder().dataFrame(predictionDataFrame).build();
        Input mlInput = MLInput.builder().algorithm(FunctionName.LINEAR_REGRESSION).inputDataset(inputDataset).build();
        MLPredictionOutput output = (MLPredictionOutput)MLEngine.predict(mlInput, model);
        DataFrame predictions = output.getPredictionResult();
        Assert.assertEquals(2, predictions.size());
    }

    @Test
    public void trainKMeans() {
        Model model = trainKMeansModel();
        Assert.assertEquals(FunctionName.KMEANS.name(), model.getName());
        Assert.assertEquals(1, model.getVersion());
        Assert.assertNotNull(model.getContent());
    }

    @Test
    public void trainLinearRegression() {
        Model model = trainLinearRegressionModel();
        Assert.assertEquals(FunctionName.LINEAR_REGRESSION.name(), model.getName());
        Assert.assertEquals(1, model.getVersion());
        Assert.assertNotNull(model.getContent());
    }

    @Test
    public void train_NullInput() {
        exceptionRule.expect(IllegalArgumentException.class);
        exceptionRule.expectMessage("Input should not be null");
        FunctionName algoName = FunctionName.LINEAR_REGRESSION;
        try (MockedStatic<MLEngineClassLoader> loader = Mockito.mockStatic(MLEngineClassLoader.class)) {
            loader.when(() -> MLEngineClassLoader.initInstance(algoName, null, MLAlgoParams.class)).thenReturn(null);
            MLEngine.train(null);
        }
    }

    @Test
    public void train_NullDataFrame() {
        exceptionRule.expect(IllegalArgumentException.class);
        exceptionRule.expectMessage("Input data frame should not be null or empty");
        FunctionName algoName = FunctionName.LINEAR_REGRESSION;
        try (MockedStatic<MLEngineClassLoader> loader = Mockito.mockStatic(MLEngineClassLoader.class)) {
            loader.when(() -> MLEngineClassLoader.initInstance(algoName, null, MLAlgoParams.class)).thenReturn(null);
            MLEngine.train(MLInput.builder().algorithm(algoName).build());
        }
    }

    @Test
    public void train_EmptyDataFrame() {
        exceptionRule.expect(IllegalArgumentException.class);
        exceptionRule.expectMessage("Input data frame should not be null or empty");
        FunctionName algoName = FunctionName.LINEAR_REGRESSION;
        try (MockedStatic<MLEngineClassLoader> loader = Mockito.mockStatic(MLEngineClassLoader.class)) {
            loader.when(() -> MLEngineClassLoader.initInstance(algoName, null, MLAlgoParams.class)).thenReturn(null);
            MLInputDataset inputDataset = DataFrameInputDataset.builder().dataFrame(constructKMeansDataFrame(0)).build();
            MLEngine.train(MLInput.builder().algorithm(algoName).inputDataset(inputDataset).build());
        }
    }

    @Test
    public void train_UnsupportedAlgorithm() {
        exceptionRule.expect(IllegalArgumentException.class);
        exceptionRule.expectMessage("Unsupported algorithm: LINEAR_REGRESSION");
        FunctionName algoName = FunctionName.LINEAR_REGRESSION;
        try (MockedStatic<MLEngineClassLoader> loader = Mockito.mockStatic(MLEngineClassLoader.class)) {
            loader.when(() -> MLEngineClassLoader.initInstance(algoName, null, MLAlgoParams.class)).thenReturn(null);
            MLInputDataset inputDataset = DataFrameInputDataset.builder().dataFrame(constructKMeansDataFrame(10)).build();
            MLEngine.train(MLInput.builder().algorithm(algoName).inputDataset(inputDataset).build());
        }
    }

    @Test
    public void predictNullInput() {
        exceptionRule.expect(IllegalArgumentException.class);
        exceptionRule.expectMessage("Input should not be null");
        MLEngine.predict(null, null);
    }

    @Test
    public void predictWithoutAlgoName() {
        exceptionRule.expect(IllegalArgumentException.class);
        exceptionRule.expectMessage("algorithm can't be null");
        MLInputDataset inputDataset = DataFrameInputDataset.builder().dataFrame(constructKMeansDataFrame(10)).build();
        Input mlInput = MLInput.builder().inputDataset(inputDataset).build();
        MLEngine.predict(mlInput, null);
    }

    @Test
    public void predictWithoutModel() {
        exceptionRule.expect(IllegalArgumentException.class);
        exceptionRule.expectMessage("No model found for linear regression prediction.");
        MLInputDataset inputDataset = DataFrameInputDataset.builder().dataFrame(constructLinearRegressionPredictionDataFrame()).build();
        Input mlInput = MLInput.builder().algorithm(FunctionName.LINEAR_REGRESSION).inputDataset(inputDataset).build();
        MLEngine.predict(mlInput, null);
    }

    @Test
    public void predictUnsupportedAlgorithm() {
        exceptionRule.expect(IllegalArgumentException.class);
        exceptionRule.expectMessage("Unsupported algorithm: LINEAR_REGRESSION");
        FunctionName algoName = FunctionName.LINEAR_REGRESSION;
        try (MockedStatic<MLEngineClassLoader> loader = Mockito.mockStatic(MLEngineClassLoader.class)) {
            loader.when(() -> MLEngineClassLoader.initInstance(algoName, null, MLAlgoParams.class)).thenReturn(null);
            MLInputDataset inputDataset = DataFrameInputDataset.builder().dataFrame(constructLinearRegressionPredictionDataFrame()).build();
            Input mlInput = MLInput.builder().algorithm(algoName).inputDataset(inputDataset).build();
            MLEngine.predict(mlInput, null);
        }
    }

    @Test
    public void testAD() {

        List<double[]> fourclassData = new ArrayList<>();
        fourclassData.add(new double[]{ 0.747253, 0.894737 });
        fourclassData.add(new double[]{ 0.472527, -0.0643275 });
        fourclassData.add(new double[]{ 0.362637, 0.789474 });
        fourclassData.add(new double[]{ -0.692308, 0.836257 });
        fourclassData.add(new double[]{ 0.153846, 0.274854 });
        fourclassData.add(new double[]{ 0.10989, -0.929825 });
        fourclassData.add(new double[]{ -0.032967, -0.00584795 });
        fourclassData.add(new double[]{ -0.747253, -0.602339 });
        fourclassData.add(new double[]{ -0.582418, 0.847953 });
        fourclassData.add(new double[]{ -0.351648, -0.602339 });
        fourclassData.add(new double[]{ -0.725275, -0.438596 });
        fourclassData.add(new double[]{ -0.824176, -0.48538 });
        fourclassData.add(new double[]{ -0.10989, 0.380117 });
        fourclassData.add(new double[]{ -0.373626, 0.450292 });
        fourclassData.add(new double[]{ 0.934066, 0.555556 });
        fourclassData.add(new double[]{ 0.417582, 0.80117 });
        fourclassData.add(new double[]{ -0.186813, -0.00584795 });
        fourclassData.add(new double[]{ -0.296703, 0.497076 });
        fourclassData.add(new double[]{ 0.153846, -0.0994152 });
        fourclassData.add(new double[]{ 0.450549, -0.0760234 });
        fourclassData.add(new double[]{ -0.769231, -0.415205 });
        fourclassData.add(new double[]{ -0.010989, -0.0292398 });
        fourclassData.add(new double[]{ -0.142857, -0.578947 });
        fourclassData.add(new double[]{ -0.0879121, -0.871345 });
        fourclassData.add(new double[]{ -0.032967, 0.368421 });
        fourclassData.add(new double[]{ 0.758242, 0.859649 });
        fourclassData.add(new double[]{ 0.736264, 0.80117 });
        fourclassData.add(new double[]{ -0.571429, 0.22807 });
        fourclassData.add(new double[]{ 0.351648, 0.169591 });
        fourclassData.add(new double[]{ 0.846154, 0.836257 });
        fourclassData.add(new double[]{ -0.648352, 0.695906 });
        fourclassData.add(new double[]{ 0.186813, 0.309942 });
        fourclassData.add(new double[]{ -0.901099, 0.239766 });
        fourclassData.add(new double[]{ -0.406593, -0.0994152 });
        fourclassData.add(new double[]{ -0.637363, 0.0877193 });
        fourclassData.add(new double[]{ 0.21978, -0.812865 });
        fourclassData.add(new double[]{ -0.78022, -0.461988 });
        fourclassData.add(new double[]{ 0.395604, 0.0175439 });
        fourclassData.add(new double[]{ -0.956044, 0.473684 });
        fourclassData.add(new double[]{ 0.296703, 0.216374 });
        fourclassData.add(new double[]{ 0.10989, -0.707602 });
        fourclassData.add(new double[]{ -0.373626, 0.590643 });
        fourclassData.add(new double[]{ 0.0659341, -0.730994 });
        fourclassData.add(new double[]{ -0.56044, 0.660819 });
        fourclassData.add(new double[]{ -0.813187, 0.356725 });
        fourclassData.add(new double[]{ 0.142857, -0.836257 });
        fourclassData.add(new double[]{ -0.32967, 0.578947 });
        fourclassData.add(new double[]{ -0.989011, 0.181287 });
        fourclassData.add(new double[]{ -0.648352, -0.157895 });
        fourclassData.add(new double[]{ -0.296703, -0.567251 });
        fourclassData.add(new double[]{ 0.384615, -0.157895 });
        fourclassData.add(new double[]{ -0.208791, -0.777778 });
        fourclassData.add(new double[]{ -0.395604, 0.578947 });
        fourclassData.add(new double[]{ 0.835165, 0.660819 });
        fourclassData.add(new double[]{ -0.637363, 0.111111 });
        fourclassData.add(new double[]{ -0.549451, 0.719298 });
        fourclassData.add(new double[]{ 0.043956, -0.777778 });
        fourclassData.add(new double[]{ -0.010989, -0.672515 });
        fourclassData.add(new double[]{ 0.021978, -0.0760234 });
        fourclassData.add(new double[]{ 0.384615, 0.789474 });
        fourclassData.add(new double[]{ -0.868132, 0.438596 });
        fourclassData.add(new double[]{ -0.186813, -0.730994 });
        fourclassData.add(new double[]{ -0.373626, 0.146199 });
        fourclassData.add(new double[]{ -0.527473, 0.80117 });
        fourclassData.add(new double[]{ -0.32967, -0.637427 });
        fourclassData.add(new double[]{ 0.417582, -0.192982 });
        fourclassData.add(new double[]{ 0.153846, 0.00584795 });
        fourclassData.add(new double[]{ 0.241758, -0.824561 });
        fourclassData.add(new double[]{ -0.230769, 0.380117 });
        fourclassData.add(new double[]{ -0.10989, -0.847953 });
        fourclassData.add(new double[]{ -0.659341, -0.0409357 });
        fourclassData.add(new double[]{ 0.868132, 0.695906 });
        fourclassData.add(new double[]{ 0.0989011, -0.0877193 });
        fourclassData.add(new double[]{ -0.714286, 0.122807 });
        fourclassData.add(new double[]{ 0.252747, 0.356725 });
        fourclassData.add(new double[]{ -0.846154, 0.380117 });
        fourclassData.add(new double[]{ -0.406593, 0.134503 });
        fourclassData.add(new double[]{ -0.56044, -0.111111 });
        fourclassData.add(new double[]{ -0.318681, 0.473684 });
        fourclassData.add(new double[]{ -0.648352, 0.707602 });
        fourclassData.add(new double[]{ -0.516484, -0.812865 });
        fourclassData.add(new double[]{ 0.0659341, 0.28655 });
        fourclassData.add(new double[]{ 0.791209, 0.918129 });
        fourclassData.add(new double[]{ 0.934066, 0.602339 });
        fourclassData.add(new double[]{ -0.802198, -0.602339 });
        fourclassData.add(new double[]{ 0.791209, 0.871345 });
        fourclassData.add(new double[]{ 0.505495, 0.929825 });
        fourclassData.add(new double[]{ -0.472527, -0.74269 });
        fourclassData.add(new double[]{ -0.142857, -0.707602 });
        fourclassData.add(new double[]{ 0.362637, -0.239766 });
        fourclassData.add(new double[]{ -0.582418, 0.239766 });
        fourclassData.add(new double[]{ 0.21978, -0.964912 });
        fourclassData.add(new double[]{ -0.362637, 0.403509 });
        fourclassData.add(new double[]{ -0.637363, -0.625731 });
        fourclassData.add(new double[]{ 0.351648, 0.146199 });
        fourclassData.add(new double[]{ -0.021978, 0.426901 });
        fourclassData.add(new double[]{ -0.230769, 0.508772 });
        fourclassData.add(new double[]{ -0.758242, -0.438596 });
        fourclassData.add(new double[]{ -0.505495, -0.836257 });
        fourclassData.add(new double[]{ 0.373626, -0.0760234 });
        fourclassData.add(new double[]{ 0.417582, 0.0643275 });
        fourclassData.add(new double[]{ -0.78022, -0.181287 });
        fourclassData.add(new double[]{ 0.0989011, -0.836257 });
        fourclassData.add(new double[]{ -0.714286, 0.906433 });
        fourclassData.add(new double[]{ -0.945055, 0.649123 });
        fourclassData.add(new double[]{ 0.703297, 0.812865 });
        fourclassData.add(new double[]{ -0.571429, -0.0760234 });
        fourclassData.add(new double[]{ -0.208791, 0.48538 });
        fourclassData.add(new double[]{ -0.879121, 0.403509 });
        fourclassData.add(new double[]{ 0.868132, 0.614035 });
        fourclassData.add(new double[]{ -0.0879121, -0.695906 });
        fourclassData.add(new double[]{ -0.406593, 0.532164 });
        fourclassData.add(new double[]{ -0.142857, -0.122807 });
        fourclassData.add(new double[]{ -0.142857, -0.0526316 });
        fourclassData.add(new double[]{ -0.538462, 0.111111 });
        fourclassData.add(new double[]{ 0.208791, 0.380117 });
        fourclassData.add(new double[]{ -0.615385, -0.134503 });
        fourclassData.add(new double[]{ -0.21978, -0.122807 });
        fourclassData.add(new double[]{ -0.736264, -0.0877193 });
        fourclassData.add(new double[]{ -0.461538, -0.0409357 });
        fourclassData.add(new double[]{ 0.769231, 0.789474 });
        fourclassData.add(new double[]{ 0.0659341, -0.94152 });
        fourclassData.add(new double[]{ -0.802198, 0.906433 });
        fourclassData.add(new double[]{ 0.384615, 0.94152 });
        fourclassData.add(new double[]{ -0.252747, 0.321637 });
        fourclassData.add(new double[]{ -0.747253, -0.532164 });
        fourclassData.add(new double[]{ -0.21978, 0.309942 });
        fourclassData.add(new double[]{ 0.516484, -0.146199 });
        fourclassData.add(new double[]{ -0.274725, -0.766082 });
        fourclassData.add(new double[]{ -0.604396, 0.847953 });
        fourclassData.add(new double[]{ -0.681319, 0.637427 });
        fourclassData.add(new double[]{ 0.527473, 0.964912 });
        fourclassData.add(new double[]{ -0.417582, -0.660819 });
        fourclassData.add(new double[]{ -0.538462, 0.22807 });
        fourclassData.add(new double[]{ -0.582418, 0.181287 });
        fourclassData.add(new double[]{ -0.703297, -0.74269 });
        fourclassData.add(new double[]{ 0.032967, -0.134503 });
        fourclassData.add(new double[]{ -0.351648, -0.567251 });
        fourclassData.add(new double[]{ -0.626374, -0.707602 });
        fourclassData.add(new double[]{ -0.318681, -0.719298 });
        fourclassData.add(new double[]{ -0.692308, -0.754386 });
        fourclassData.add(new double[]{ -0.538462, -0.812865 });
        fourclassData.add(new double[]{ -0.626374, 0.707602 });
        fourclassData.add(new double[]{ -0.67033, 0.789474 });
        fourclassData.add(new double[]{ -0.0879121, 0.0643275 });
        fourclassData.add(new double[]{ -0.747253, 0.111111 });
        fourclassData.add(new double[]{ -0.604396, 0.94152 });
        fourclassData.add(new double[]{ -0.384615, -0.660819 });
        fourclassData.add(new double[]{ -0.164835, -0.730994 });
        fourclassData.add(new double[]{ -0.340659, -0.730994 });
        fourclassData.add(new double[]{ -0.384615, 0.48538 });
        fourclassData.add(new double[]{ 0.340659, 0.929825 });
        fourclassData.add(new double[]{ 0.67033, 0.847953 });
        fourclassData.add(new double[]{ -0.351648, -0.508772 });
        fourclassData.add(new double[]{ 0.0769231, 0.274854 });
        fourclassData.add(new double[]{ -0.835165, -0.54386 });
        fourclassData.add(new double[]{ 0.032967, -0.94152 });
        fourclassData.add(new double[]{ 0.0769231, -0.660819 });
        fourclassData.add(new double[]{ -0.406593, -0.730994 });
        fourclassData.add(new double[]{ -0.318681, -0.54386 });
        fourclassData.add(new double[]{ -0.824176, -0.520468 });
        fourclassData.add(new double[]{ -0.307692, -0.695906 });
        fourclassData.add(new double[]{ -0.285714, 0.567251 });
        fourclassData.add(new double[]{ 0.0549451, 0.415205 });
        fourclassData.add(new double[]{ 0.582418, 0.976608 });
        fourclassData.add(new double[]{ -0.527473, -0.730994 });
        fourclassData.add(new double[]{ -0.89011, 0.695906 });
        fourclassData.add(new double[]{ 0.043956, -0.918129 });
        fourclassData.add(new double[]{ -0.021978, -0.122807 });
        fourclassData.add(new double[]{ 0.021978, -0.204678 });
        fourclassData.add(new double[]{ 0.483516, -0.0877193 });
        fourclassData.add(new double[]{ -0.747253, 0.0643275 });
        fourclassData.add(new double[]{ -1.0, 0.321637 });
        fourclassData.add(new double[]{ -0.263736, -0.157895 });
        fourclassData.add(new double[]{ 0.549451, 0.883041 });
        fourclassData.add(new double[]{ 0.648352, 0.812865 });
        fourclassData.add(new double[]{ -0.32967, -0.146199 });
        fourclassData.add(new double[]{ -0.472527, -0.754386 });
        fourclassData.add(new double[]{ -0.703297, 0.0760234 });
        fourclassData.add(new double[]{ 0.0769231, 0.00584795 });
        fourclassData.add(new double[]{ 0.362637, 0.812865 });
        fourclassData.add(new double[]{ -0.604396, 0.602339 });
        fourclassData.add(new double[]{ -0.351648, -0.122807 });
        fourclassData.add(new double[]{ 0.527473, -0.111111 });
        fourclassData.add(new double[]{ 0.494505, 0.859649 });
        fourclassData.add(new double[]{ -0.186813, -0.0526316 });
        fourclassData.add(new double[]{ -0.648352, 0.602339 });
        fourclassData.add(new double[]{ -0.395604, -0.730994 });
        fourclassData.add(new double[]{ 0.208791, 0.0994152 });
        fourclassData.add(new double[]{ 0.461538, 0.859649 });
        fourclassData.add(new double[]{ -0.615385, -0.625731 });
        fourclassData.add(new double[]{ 0.692308, 0.906433 });
        fourclassData.add(new double[]{ -0.626374, -0.169591 });
        fourclassData.add(new double[]{ -0.758242, -0.508772 });
        fourclassData.add(new double[]{ 0.373626, 0.894737 });
        fourclassData.add(new double[]{ -0.472527, 0.0760234 });
        fourclassData.add(new double[]{ 0.318681, 0.0526316 });
        fourclassData.add(new double[]{ -0.681319, 0.906433 });
        fourclassData.add(new double[]{ 0.340659, -0.216374 });
        fourclassData.add(new double[]{ -0.604396, 0.719298 });
        fourclassData.add(new double[]{ 0.626374, 0.871345 });
        fourclassData.add(new double[]{ -0.571429, 0.134503 });
        fourclassData.add(new double[]{ 0.153846, -0.192982 });
        fourclassData.add(new double[]{ -0.879121, 0.695906 });
        fourclassData.add(new double[]{ -0.648352, -0.169591 });
        fourclassData.add(new double[]{ 0.230769, 0.251462 });
        fourclassData.add(new double[]{ -0.241758, -0.625731 });
        fourclassData.add(new double[]{ -0.912088, 0.508772 });
        fourclassData.add(new double[]{ 0.32967, 0.871345 });
        fourclassData.add(new double[]{ -0.835165, 0.450292 });
        fourclassData.add(new double[]{ -0.472527, -0.590643 });
        fourclassData.add(new double[]{ -0.648352, 0.953216 });
        fourclassData.add(new double[]{ -0.802198, -0.660819 });
        fourclassData.add(new double[]{ -0.362637, 0.0409357 });
        fourclassData.add(new double[]{ -0.956044, 0.649123 });
        fourclassData.add(new double[]{ -0.472527, -0.00584795 });
        fourclassData.add(new double[]{ -0.197802, -0.0175439 });
        fourclassData.add(new double[]{ 0.0879121, -0.672515 });
        fourclassData.add(new double[]{ 0.857143, 0.777778 });
        fourclassData.add(new double[]{ -0.835165, -0.497076 });
        fourclassData.add(new double[]{ -0.582418, 0.660819 });
        fourclassData.add(new double[]{ -0.67033, 0.637427 });
        fourclassData.add(new double[]{ -0.351648, -0.0994152 });
        fourclassData.add(new double[]{ 0.032967, 0.450292 });
        fourclassData.add(new double[]{ -0.791209, -0.497076 });
        fourclassData.add(new double[]{ -0.582418, 0.216374 });
        fourclassData.add(new double[]{ 0.285714, 0.216374 });
        fourclassData.add(new double[]{ 0.153846, -0.0877193 });
        fourclassData.add(new double[]{ -0.010989, -0.0643275 });
        fourclassData.add(new double[]{ 0.285714, 0.111111 });
        fourclassData.add(new double[]{ -0.802198, -0.134503 });
        fourclassData.add(new double[]{ -0.021978, -0.111111 });
        fourclassData.add(new double[]{ 0.032967, 0.251462 });
        fourclassData.add(new double[]{ -0.835165, -0.403509 });
        fourclassData.add(new double[]{ -0.571429, 0.192982 });
        fourclassData.add(new double[]{ -0.637363, -0.169591 });
        fourclassData.add(new double[]{ -0.307692, -0.0760234 });
        fourclassData.add(new double[]{ -0.318681, -0.157895 });
        fourclassData.add(new double[]{ 0.483516, 0.906433 });
        fourclassData.add(new double[]{ -0.428571, 0.122807 });
        fourclassData.add(new double[]{ -0.0659341, -0.555556 });
        fourclassData.add(new double[]{ -0.549451, 0.22807 });
        fourclassData.add(new double[]{ -0.384615, -0.730994 });
        fourclassData.add(new double[]{ -0.362637, -0.789474 });
        fourclassData.add(new double[]{ -0.461538, 0.660819 });
        fourclassData.add(new double[]{ 0.758242, 0.824561 });
        fourclassData.add(new double[]{ 0.813187, 0.894737 });
        fourclassData.add(new double[]{ 0.175824, -0.964912 });
        fourclassData.add(new double[]{ -0.538462, 0.251462 });
        fourclassData.add(new double[]{ -0.516484, -0.0643275 });
        fourclassData.add(new double[]{ 0.0549451, -0.74269 });
        fourclassData.add(new double[]{ 0.021978, 0.0526316 });
        fourclassData.add(new double[]{ -0.0989011, -0.625731 });
        fourclassData.add(new double[]{ 0.604396, 0.929825 });
        fourclassData.add(new double[]{ 0.230769, 0.169591 });
        fourclassData.add(new double[]{ -0.428571, 0.602339 });
        fourclassData.add(new double[]{ -0.043956, -0.918129 });
        fourclassData.add(new double[]{ -0.230769, -0.637427 });
        fourclassData.add(new double[]{ -0.307692, -0.567251 });
        fourclassData.add(new double[]{ -0.626374, -0.0877193 });
        fourclassData.add(new double[]{ -0.791209, -0.54386 });
        fourclassData.add(new double[]{ -0.340659, 0.0877193 });
        fourclassData.add(new double[]{ -0.0989011, 0.0409357 });
        fourclassData.add(new double[]{ 0.945055, 0.707602 });
        fourclassData.add(new double[]{ 0.186813, 0.380117 });
        fourclassData.add(new double[]{ 0.0879121, -1.0 });
        fourclassData.add(new double[]{ 0.0549451, 0.239766 });
        fourclassData.add(new double[]{ -0.373626, -0.625731 });
        fourclassData.add(new double[]{ -0.901099, 1.0 });
        fourclassData.add(new double[]{ -0.604396, 0.80117 });
        fourclassData.add(new double[]{ -0.252747, 0.309942 });
        fourclassData.add(new double[]{ -0.274725, -0.789474 });
        fourclassData.add(new double[]{ 0.208791, 0.28655 });
        fourclassData.add(new double[]{ 0.472527, -0.0877193 });
        fourclassData.add(new double[]{ -0.56044, -0.0409357 });
        fourclassData.add(new double[]{ -0.153846, -0.730994 });
        fourclassData.add(new double[]{ -0.450549, -0.0409357 });
        fourclassData.add(new double[]{ 0.285714, 0.309942 });
        fourclassData.add(new double[]{ 0.21978, -0.777778 });
        fourclassData.add(new double[]{ -0.395604, -0.74269 });
        fourclassData.add(new double[]{ 0.934066, 0.567251 });
        fourclassData.add(new double[]{ 0.120879, 0.274854 });
        fourclassData.add(new double[]{ -0.67033, -0.298246 });
        fourclassData.add(new double[]{ -0.659341, -0.0643275 });
        fourclassData.add(new double[]{ -0.373626, -0.0643275 });
        fourclassData.add(new double[]{ -0.285714, 0.426901 });
        fourclassData.add(new double[]{ 0.32967, 0.146199 });
        fourclassData.add(new double[]{ -0.428571, 0.426901 });
        fourclassData.add(new double[]{ 0.032967, -0.649123 });
        fourclassData.add(new double[]{ -0.626374, -0.497076 });
        fourclassData.add(new double[]{ -0.769231, -0.497076 });
        fourclassData.add(new double[]{ -0.593407, -0.730994 });
        fourclassData.add(new double[]{ -0.582418, -0.824561 });
        fourclassData.add(new double[]{ -0.0659341, -0.719298 });
        fourclassData.add(new double[]{ -0.604396, 0.134503 });
        fourclassData.add(new double[]{ 0.373626, 0.0526316 });
        fourclassData.add(new double[]{ -0.659341, -0.508772 });
        fourclassData.add(new double[]{ -0.692308, -0.80117 });
        fourclassData.add(new double[]{ -0.978022, 0.181287 });
        fourclassData.add(new double[]{ -0.362637, -0.0175439 });
        fourclassData.add(new double[]{ -0.32967, 0.508772 });
        fourclassData.add(new double[]{ -0.010989, -0.719298 });
        fourclassData.add(new double[]{ 0.164835, -0.988304 });
        fourclassData.add(new double[]{ -0.56044, 0.00584795 });
        fourclassData.add(new double[]{ -0.0989011, -0.777778 });
        fourclassData.add(new double[]{ -0.659341, 0.146199 });
        fourclassData.add(new double[]{ -0.593407, 0.590643 });
        fourclassData.add(new double[]{ -0.56044, 0.649123 });
        fourclassData.add(new double[]{ -0.032967, 0.321637 });
        fourclassData.add(new double[]{ -0.791209, 0.0526316 });
        fourclassData.add(new double[]{ -0.0879121, -0.894737 });
        fourclassData.add(new double[]{ 0.857143, 0.836257 });
        fourclassData.add(new double[]{ -0.549451, 0.859649 });
        fourclassData.add(new double[]{ -0.648352, -0.426901 });
        fourclassData.add(new double[]{ -0.230769, -0.146199 });
        fourclassData.add(new double[]{ -0.593407, 0.263158 });
        fourclassData.add(new double[]{ -0.78022, 0.0760234 });
        fourclassData.add(new double[]{ -0.032967, 0.415205 });
        fourclassData.add(new double[]{ -0.824176, 0.532164 });
        fourclassData.add(new double[]{ 0.978022, 0.730994 });
        fourclassData.add(new double[]{ 0.10989, -0.74269 });
        fourclassData.add(new double[]{ -0.318681, 0.391813 });
        fourclassData.add(new double[]{ -0.835165, -0.216374 });
        fourclassData.add(new double[]{ -0.461538, -0.707602 });
        fourclassData.add(new double[]{ -0.0659341, -0.0643275 });
        fourclassData.add(new double[]{ -0.725275, 0.0643275 });
        fourclassData.add(new double[]{ -0.538462, -0.0526316 });
        fourclassData.add(new double[]{ -0.483516, -0.812865 });
        fourclassData.add(new double[]{ 0.901099, 0.684211 });
        fourclassData.add(new double[]{ 0.274725, 0.28655 });
        fourclassData.add(new double[]{ 0.89011, 0.871345 });
        fourclassData.add(new double[]{ -0.714286, 0.00584795 });
        fourclassData.add(new double[]{ 0.274725, 0.22807 });
        fourclassData.add(new double[]{ -0.703297, 0.847953 });
        fourclassData.add(new double[]{ -0.0989011, 0.0526316 });
        fourclassData.add(new double[]{ -0.758242, 0.0994152 });
        fourclassData.add(new double[]{ -0.615385, 0.22807 });
        fourclassData.add(new double[]{ -0.67033, 0.754386 });
        fourclassData.add(new double[]{ -0.791209, -0.461988 });
        fourclassData.add(new double[]{ -0.120879, -0.0994152 });
        fourclassData.add(new double[]{ -0.637363, 0.871345 });
        fourclassData.add(new double[]{ 0.494505, 0.976608 });
        fourclassData.add(new double[]{ -0.538462, 0.730994 });
        fourclassData.add(new double[]{ -0.912088, 0.871345 });
        fourclassData.add(new double[]{ 0.153846, -0.0409357 });
        fourclassData.add(new double[]{ 0.0659341, -0.146199 });
        fourclassData.add(new double[]{ -0.230769, 0.497076 });
        fourclassData.add(new double[]{ 0.351648, 0.789474 });
        fourclassData.add(new double[]{ 0.824176, 0.777778 });
        fourclassData.add(new double[]{ 0.0659341, -0.0526316 });
        fourclassData.add(new double[]{ -0.549451, 0.0877193 });
        fourclassData.add(new double[]{ -0.912088, 0.157895 });
        fourclassData.add(new double[]{ 1.0, 0.578947 });
        fourclassData.add(new double[]{ 0.0989011, 0.380117 });
        fourclassData.add(new double[]{ -0.637363, -0.0643275 });
        fourclassData.add(new double[]{ 0.461538, -0.146199 });
        fourclassData.add(new double[]{ 0.351648, 0.157895 });
        fourclassData.add(new double[]{ -0.43956, 0.461988 });
        fourclassData.add(new double[]{ -0.296703, 0.368421 });
        fourclassData.add(new double[]{ -0.0659341, 0.380117 });
        fourclassData.add(new double[]{ -0.351648, 0.146199 });
        fourclassData.add(new double[]{ -0.615385, 0.74269 });
        fourclassData.add(new double[]{ -0.923077, 0.695906 });
        fourclassData.add(new double[]{ -0.67033, -0.707602 });
        fourclassData.add(new double[]{ -0.285714, 0.403509 });
        fourclassData.add(new double[]{ -0.637363, 0.766082 });
        fourclassData.add(new double[]{ -0.527473, 0.204678 });
        fourclassData.add(new double[]{ -0.703297, 0.0409357 });
        fourclassData.add(new double[]{ -0.197802, 0.403509 });
        fourclassData.add(new double[]{ 0.153846, 0.146199 });
        fourclassData.add(new double[]{ -0.626374, -0.48538 });
        fourclassData.add(new double[]{ 0.505495, -0.181287 });
        fourclassData.add(new double[]{ -0.703297, 0.766082 });
        fourclassData.add(new double[]{ -0.32967, -0.74269 });
        fourclassData.add(new double[]{ -0.417582, 0.146199 });
        fourclassData.add(new double[]{ 0.021978, 0.403509 });
        fourclassData.add(new double[]{ -0.593407, 0.578947 });
        fourclassData.add(new double[]{ 0.175824, -1.0 });
        fourclassData.add(new double[]{ 0.461538, 0.146199 });
        fourclassData.add(new double[]{ -0.362637, 0.625731 });
        fourclassData.add(new double[]{ 0.21978, 0.894737 });
        fourclassData.add(new double[]{ -0.67033, -0.614035 });
        fourclassData.add(new double[]{ -0.263736, 0.567251 });
        fourclassData.add(new double[]{ 0.417582, 0.789474 });
        fourclassData.add(new double[]{ 0.571429, 0.953216 });
        fourclassData.add(new double[]{ -0.032967, -0.169591 });
        fourclassData.add(new double[]{ -0.384615, -0.0175439 });
        fourclassData.add(new double[]{ 0.285714, 0.80117 });
        fourclassData.add(new double[]{ -0.626374, -0.672515 });
        fourclassData.add(new double[]{ -0.604396, 0.0292398 });
        fourclassData.add(new double[]{ -0.197802, 0.0292398 });
        fourclassData.add(new double[]{ -0.516484, 0.263158 });
        fourclassData.add(new double[]{ -0.637363, 0.00584795 });
        fourclassData.add(new double[]{ -0.868132, 0.614035 });
        fourclassData.add(new double[]{ -0.417582, 0.111111 });
        fourclassData.add(new double[]{ -0.461538, -0.590643 });
        fourclassData.add(new double[]{ 0.10989, -0.146199 });
        fourclassData.add(new double[]{ -0.582418, 0.649123 });
        fourclassData.add(new double[]{ 0.142857, -0.730994 });
        fourclassData.add(new double[]{ -0.406593, 0.590643 });
        fourclassData.add(new double[]{ 0.10989, -0.789474 });
        fourclassData.add(new double[]{ -0.637363, 0.672515 });
        fourclassData.add(new double[]{ -0.56044, 0.74269 });
        fourclassData.add(new double[]{ -0.648352, 0.824561 });
        fourclassData.add(new double[]{ 0.923077, 0.707602 });
        fourclassData.add(new double[]{ -0.307692, 0.0760234 });
        fourclassData.add(new double[]{ 0.351648, -0.146199 });
        fourclassData.add(new double[]{ -0.538462, -0.695906 });
        fourclassData.add(new double[]{ -0.538462, 0.614035 });
        fourclassData.add(new double[]{ -0.472527, -0.0526316 });
        fourclassData.add(new double[]{ -0.89011, 0.263158 });
        fourclassData.add(new double[]{ -0.483516, 0.239766 });
        fourclassData.add(new double[]{ -0.296703, -0.0292398 });
        fourclassData.add(new double[]{ -0.802198, 0.602339 });
        fourclassData.add(new double[]{ -0.131868, -0.0409357 });
        fourclassData.add(new double[]{ -0.945055, 0.54386 });
        fourclassData.add(new double[]{ 0.296703, 0.239766 });
        fourclassData.add(new double[]{ -0.318681, 0.508772 });
        fourclassData.add(new double[]{ -0.252747, -0.614035 });
        fourclassData.add(new double[]{ -0.131868, 0.0409357 });
        fourclassData.add(new double[]{ -0.78022, -0.730994 });
        fourclassData.add(new double[]{ -0.417582, 0.590643 });
        fourclassData.add(new double[]{ 0.351648, 0.80117 });
        fourclassData.add(new double[]{ 0.615385, 0.871345 });
        fourclassData.add(new double[]{ -0.956044, 0.239766 });
        fourclassData.add(new double[]{ 0.153846, 0.309942 });
        fourclassData.add(new double[]{ -0.89011, 0.497076 });
        fourclassData.add(new double[]{ 0.362637, 0.0175439 });
        fourclassData.add(new double[]{ -0.912088, 0.309942 });
        fourclassData.add(new double[]{ -0.010989, 0.0292398 });
        fourclassData.add(new double[]{ -0.0549451, -0.0643275 });
        fourclassData.add(new double[]{ -0.417582, -0.602339 });
        fourclassData.add(new double[]{ -0.428571, -0.672515 });
        fourclassData.add(new double[]{ -0.186813, -0.719298 });
        fourclassData.add(new double[]{ 0.538462, 0.906433 });
        fourclassData.add(new double[]{ 0.901099, 0.602339 });
        fourclassData.add(new double[]{ 0.0549451, -0.660819 });
        fourclassData.add(new double[]{ 0.868132, 0.672515 });
        fourclassData.add(new double[]{ -0.021978, 0.356725 });
        fourclassData.add(new double[]{ -0.395604, 0.216374 });
        fourclassData.add(new double[]{ -0.0879121, 0.426901 });
        fourclassData.add(new double[]{ 0.32967, 0.28655 });
        fourclassData.add(new double[]{ 0.021978, 0.0409357 });
        fourclassData.add(new double[]{ -0.131868, -0.812865 });
        fourclassData.add(new double[]{ 0.153846, 0.169591 });
        fourclassData.add(new double[]{ -0.197802, -0.00584795 });
        fourclassData.add(new double[]{ 0.208791, 0.157895 });
        fourclassData.add(new double[]{ 0.241758, 0.894737 });
        fourclassData.add(new double[]{ -0.296703, -0.777778 });
        fourclassData.add(new double[]{ -0.0989011, -0.660819 });
        fourclassData.add(new double[]{ -0.604396, -0.567251 });
        fourclassData.add(new double[]{ -0.010989, 0.0643275 });
        fourclassData.add(new double[]{ -0.824176, -0.391813 });
        fourclassData.add(new double[]{ -0.241758, 0.0175439 });
        fourclassData.add(new double[]{ -0.021978, -0.146199 });
        fourclassData.add(new double[]{ 0.197802, 0.263158 });
        fourclassData.add(new double[]{ -0.307692, -0.111111 });
        fourclassData.add(new double[]{ -0.681319, -0.625731 });
        fourclassData.add(new double[]{ -0.604396, 0.251462 });
        fourclassData.add(new double[]{ -0.978022, 0.438596 });
        fourclassData.add(new double[]{ -0.043956, -0.157895 });
        fourclassData.add(new double[]{ -0.494505, 0.649123 });
        fourclassData.add(new double[]{ 0.043956, -0.0994152 });
        fourclassData.add(new double[]{ -0.835165, -0.0994152 });
        fourclassData.add(new double[]{ -0.0549451, -0.192982 });
        fourclassData.add(new double[]{ 0.571429, 0.894737 });
        fourclassData.add(new double[]{ -0.131868, -0.847953 });
        fourclassData.add(new double[]{ -0.791209, -0.274854 });
        fourclassData.add(new double[]{ -0.791209, -0.695906 });
        fourclassData.add(new double[]{ -0.527473, 0.0409357 });
        fourclassData.add(new double[]{ -0.0769231, 0.333333 });
        fourclassData.add(new double[]{ 0.89011, 0.812865 });
        fourclassData.add(new double[]{ 0.604396, 0.953216 });
        fourclassData.add(new double[]{ -0.538462, 0.146199 });
        fourclassData.add(new double[]{ -0.0989011, 0.298246 });
        fourclassData.add(new double[]{ 0.0989011, 0.192982 });
        fourclassData.add(new double[]{ -0.604396, 0.812865 });
        fourclassData.add(new double[]{ 0.67033, 0.906433 });
        fourclassData.add(new double[]{ -0.857143, 0.567251 });
        fourclassData.add(new double[]{ 0.725275, 0.906433 });
        fourclassData.add(new double[]{ -0.626374, 0.192982 });
        fourclassData.add(new double[]{ -0.021978, -0.0760234 });
        fourclassData.add(new double[]{ 0.340659, 0.812865 });
        fourclassData.add(new double[]{ -0.318681, 0.450292 });
        fourclassData.add(new double[]{ 0.450549, 0.906433 });
        fourclassData.add(new double[]{ -0.43956, -0.754386 });
        fourclassData.add(new double[]{ -0.923077, 0.309942 });
        fourclassData.add(new double[]{ -0.384615, 0.0760234 });
        fourclassData.add(new double[]{ -0.10989, -0.0526316 });
        fourclassData.add(new double[]{ 0.120879, 0.216374 });
        fourclassData.add(new double[]{ -0.912088, 0.497076 });
        fourclassData.add(new double[]{ -0.373626, -0.578947 });
        fourclassData.add(new double[]{ -0.406593, -0.555556 });
        fourclassData.add(new double[]{ 0.56044, 0.976608 });
        fourclassData.add(new double[]{ -0.736264, -0.684211 });
        fourclassData.add(new double[]{ 0.285714, 0.906433 });
        fourclassData.add(new double[]{ 0.120879, 0.239766 });
        fourclassData.add(new double[]{ -0.131868, -0.730994 });
        fourclassData.add(new double[]{ 0.494505, 0.122807 });
        fourclassData.add(new double[]{ -0.131868, 0.00584795 });
        fourclassData.add(new double[]{ -0.252747, -0.0175439 });
        fourclassData.add(new double[]{ -0.604396, 0.918129 });
        fourclassData.add(new double[]{ -0.912088, 0.649123 });
        fourclassData.add(new double[]{ -0.164835, -0.824561 });
        fourclassData.add(new double[]{ -0.879121, 0.976608 });
        fourclassData.add(new double[]{ 0.417582, -0.169591 });
        fourclassData.add(new double[]{ -0.483516, 0.0526316 });
        fourclassData.add(new double[]{ 0.230769, 0.871345 });
        fourclassData.add(new double[]{ 0.0769231, 0.251462 });
        fourclassData.add(new double[]{ -0.351648, 0.403509 });
        fourclassData.add(new double[]{ 0.32967, 0.918129 });
        fourclassData.add(new double[]{ -0.835165, 0.859649 });
        fourclassData.add(new double[]{ -0.824176, 0.918129 });
        fourclassData.add(new double[]{ -0.604396, -0.146199 });
        fourclassData.add(new double[]{ -0.824176, 0.859649 });
        fourclassData.add(new double[]{ -0.472527, -0.824561 });
        fourclassData.add(new double[]{ -0.747253, 0.0292398 });
        fourclassData.add(new double[]{ -0.142857, -0.555556 });
        fourclassData.add(new double[]{ 0.483516, 0.122807 });
        fourclassData.add(new double[]{ -0.450549, 0.450292 });
        fourclassData.add(new double[]{ -0.142857, -0.754386 });
        fourclassData.add(new double[]{ -0.604396, 0.0760234 });
        fourclassData.add(new double[]{ 0.714286, 0.918129 });
        fourclassData.add(new double[]{ -0.626374, 0.0994152 });
        fourclassData.add(new double[]{ -0.626374, 0.0877193 });
        fourclassData.add(new double[]{ -0.571429, 0.263158 });
        fourclassData.add(new double[]{ 0.142857, -0.812865 });
        fourclassData.add(new double[]{ -0.648352, -0.438596 });
        fourclassData.add(new double[]{ -0.021978, 0.345029 });
        fourclassData.add(new double[]{ -0.835165, 0.438596 });
        fourclassData.add(new double[]{ -0.197802, -0.54386 });
        fourclassData.add(new double[]{ -0.593407, 0.602339 });
        fourclassData.add(new double[]{ -0.824176, 0.508772 });
        fourclassData.add(new double[]{ 0.010989, 0.403509 });
        fourclassData.add(new double[]{ -0.78022, 1.0 });
        fourclassData.add(new double[]{ -0.857143, 0.54386 });
        fourclassData.add(new double[]{ -0.626374, -0.415205 });
        fourclassData.add(new double[]{ -0.483516, 0.450292 });
        fourclassData.add(new double[]{ 0.78022, 0.754386 });
        fourclassData.add(new double[]{ 1.0, 0.649123 });
        fourclassData.add(new double[]{ -0.010989, 0.403509 });
        fourclassData.add(new double[]{ 0.230769, -0.871345 });
        fourclassData.add(new double[]{ -0.615385, 0.847953 });
        fourclassData.add(new double[]{ 0.274725, 0.789474 });
        fourclassData.add(new double[]{ -0.043956, -0.836257 });
        fourclassData.add(new double[]{ -0.714286, -0.719298 });
        fourclassData.add(new double[]{ -0.505495, 0.0877193 });
        fourclassData.add(new double[]{ 0.428571, 0.953216 });
        fourclassData.add(new double[]{ -0.912088, 0.28655 });
        fourclassData.add(new double[]{ 0.142857, 0.00584795 });



        List<double[]> fourclassAnomalyData = new ArrayList<>();
        fourclassAnomalyData.add(new double[]{ 0.692308, -0.824561 });
        fourclassAnomalyData.add(new double[]{ 0.769231, -0.321637 });
        fourclassAnomalyData.add(new double[]{ -0.21978, -0.263158 });
        fourclassAnomalyData.add(new double[]{ 0.318681, 0.54386 });
        fourclassAnomalyData.add(new double[]{ 0.0989011, -0.309942 });
        fourclassAnomalyData.add(new double[]{ 0.912088, -0.730994 });
        fourclassAnomalyData.add(new double[]{ 0.912088, 0.181287 });
        fourclassAnomalyData.add(new double[]{ 0.615385, -0.812865 });
        fourclassAnomalyData.add(new double[]{ 0.912088, 0.0760234 });
        fourclassAnomalyData.add(new double[]{ 0.78022, -0.707602 });
        fourclassAnomalyData.add(new double[]{ 0.197802, 0.567251 });
        fourclassAnomalyData.add(new double[]{ 0.912088, -0.333333 });
        fourclassAnomalyData.add(new double[]{ 0.769231, -0.637427 });
        fourclassAnomalyData.add(new double[]{ 0.78022, -0.918129 });
        fourclassAnomalyData.add(new double[]{ 0.296703, 0.497076 });
        fourclassAnomalyData.add(new double[]{ 0.527473, -0.684211 });
        fourclassAnomalyData.add(new double[]{ 0.626374, 0.497076 });
        fourclassAnomalyData.add(new double[]{ 0.0879121, -0.368421 });
        fourclassAnomalyData.add(new double[]{ -0.296703, -0.298246 });
        fourclassAnomalyData.add(new double[]{ 0.758242, 0.345029 });
        fourclassAnomalyData.add(new double[]{ 0.912088, 0.0409357 });
        fourclassAnomalyData.add(new double[]{ 0.417582, 0.602339 });
        fourclassAnomalyData.add(new double[]{ 0.362637, -0.649123 });
        fourclassAnomalyData.add(new double[]{ 0.89011, -0.181287 });
        fourclassAnomalyData.add(new double[]{ 0.901099, -0.74269 });
        fourclassAnomalyData.add(new double[]{ 0.604396, -0.929825 });
        fourclassAnomalyData.add(new double[]{ 0.494505, -0.707602 });
        fourclassAnomalyData.add(new double[]{ 0.758242, 0.204678 });
        fourclassAnomalyData.add(new double[]{ 0.769231, -0.777778 });
        fourclassAnomalyData.add(new double[]{ 0.384615, 0.497076 });
        fourclassAnomalyData.add(new double[]{ 0.879121, -0.578947 });
        fourclassAnomalyData.add(new double[]{ 0.318681, 0.672515 });
        fourclassAnomalyData.add(new double[]{ 0.549451, 0.590643 });
        fourclassAnomalyData.add(new double[]{ 0.835165, -0.368421 });
        fourclassAnomalyData.add(new double[]{ 0.285714, -0.48538 });
        fourclassAnomalyData.add(new double[]{ 0.758242, -0.321637 });
        fourclassAnomalyData.add(new double[]{ 0.791209, -0.181287 });
        fourclassAnomalyData.add(new double[]{ 0.813187, -0.239766 });
        fourclassAnomalyData.add(new double[]{ 0.208791, -0.590643 });
        fourclassAnomalyData.add(new double[]{ 0.824176, 0.368421 });
        fourclassAnomalyData.add(new double[]{ 0.0549451, -0.497076 });
        fourclassAnomalyData.add(new double[]{ 0.252747, 0.567251 });
        fourclassAnomalyData.add(new double[]{ 0.197802, 0.637427 });
        fourclassAnomalyData.add(new double[]{ 0.230769, 0.602339 });
        fourclassAnomalyData.add(new double[]{ -0.0989011, 0.649123 });
        fourclassAnomalyData.add(new double[]{ 0.846154, -0.906433 });
        fourclassAnomalyData.add(new double[]{ 0.362637, 0.637427 });
        fourclassAnomalyData.add(new double[]{ 0.197802, -0.567251 });
        fourclassAnomalyData.add(new double[]{ -0.142857, 0.660819 });
        fourclassAnomalyData.add(new double[]{ 0.67033, 0.309942 });
        fourclassAnomalyData.add(new double[]{ 0.428571, -0.660819 });
        fourclassAnomalyData.add(new double[]{ -0.0989011, 0.754386 });
        fourclassAnomalyData.add(new double[]{ 0.384615, 0.473684 });
        fourclassAnomalyData.add(new double[]{ 0.10989, 0.520468 });
        fourclassAnomalyData.add(new double[]{ 0.318681, 0.614035 });
        fourclassAnomalyData.add(new double[]{ 0.857143, 0.368421 });
        fourclassAnomalyData.add(new double[]{ 0.791209, -0.707602 });
        fourclassAnomalyData.add(new double[]{ 0.197802, 0.649123 });
        fourclassAnomalyData.add(new double[]{ -0.351648, -0.391813 });
        fourclassAnomalyData.add(new double[]{ 0.263736, 0.532164 });
        fourclassAnomalyData.add(new double[]{ 0.758242, 0.169591 });
        fourclassAnomalyData.add(new double[]{ 0.505495, 0.461988 });
        fourclassAnomalyData.add(new double[]{ 0.197802, -0.555556 });
        fourclassAnomalyData.add(new double[]{ 0.923077, -0.80117 });
        fourclassAnomalyData.add(new double[]{ -0.043956, 0.707602 });
        fourclassAnomalyData.add(new double[]{ 0.56044, -0.730994 });
        fourclassAnomalyData.add(new double[]{ 0.241758, -0.567251 });
        fourclassAnomalyData.add(new double[]{ 0.912088, -0.356725 });
        fourclassAnomalyData.add(new double[]{ -0.175824, 0.672515 });
        fourclassAnomalyData.add(new double[]{ 0.758242, -0.169591 });
        fourclassAnomalyData.add(new double[]{ 0.648352, -0.929825 });
        fourclassAnomalyData.add(new double[]{ 0.67033, -1.0 });
        fourclassAnomalyData.add(new double[]{ 0.0659341, 0.695906 });
        fourclassAnomalyData.add(new double[]{ 0.483516, -0.789474 });
        fourclassAnomalyData.add(new double[]{ 0.043956, -0.28655 });
        fourclassAnomalyData.add(new double[]{ 0.846154, -0.988304 });
        fourclassAnomalyData.add(new double[]{ 0.32967, -0.719298 });
        fourclassAnomalyData.add(new double[]{ -0.10989, -0.298246 });
        fourclassAnomalyData.add(new double[]{ -0.175824, 0.812865 });
        fourclassAnomalyData.add(new double[]{ 0.769231, 0.0175439 });
        fourclassAnomalyData.add(new double[]{ 0.450549, -0.871345 });
        fourclassAnomalyData.add(new double[]{ 0.0879121, 0.660819 });
        fourclassAnomalyData.add(new double[]{ 0.32967, 0.567251 });
        fourclassAnomalyData.add(new double[]{ 0.593407, 0.461988 });
        fourclassAnomalyData.add(new double[]{ 0.901099, -0.824561 });
        fourclassAnomalyData.add(new double[]{ 0.21978, -0.461988 });
        fourclassAnomalyData.add(new double[]{ 0.89011, -0.415205 });
        fourclassAnomalyData.add(new double[]{ 0.0659341, -0.426901 });
        fourclassAnomalyData.add(new double[]{ 0.56044, 0.415205 });
        fourclassAnomalyData.add(new double[]{ 0.582418, 0.426901 });
        fourclassAnomalyData.add(new double[]{ -0.010989, 0.602339 });
        fourclassAnomalyData.add(new double[]{ -0.043956, -0.391813 });
        fourclassAnomalyData.add(new double[]{ 0.879121, -0.0175439 });
        fourclassAnomalyData.add(new double[]{ 0.186813, -0.520468 });
        fourclassAnomalyData.add(new double[]{ 0.725275, 0.345029 });
        fourclassAnomalyData.add(new double[]{ 0.142857, -0.555556 });
        fourclassAnomalyData.add(new double[]{ 0.747253, 0.263158 });
        fourclassAnomalyData.add(new double[]{ 0.56044, -0.684211 });
        fourclassAnomalyData.add(new double[]{ 0.450549, 0.48538 });
        fourclassAnomalyData.add(new double[]{ 0.340659, 0.625731 });
        fourclassAnomalyData.add(new double[]{ 0.10989, -0.368421 });
        fourclassAnomalyData.add(new double[]{ 0.824176, -0.298246 });
        fourclassAnomalyData.add(new double[]{ 0.175824, -0.368421 });
        fourclassAnomalyData.add(new double[]{ -0.0659341, 0.614035 });
        fourclassAnomalyData.add(new double[]{ 0.318681, 0.497076 });
        fourclassAnomalyData.add(new double[]{ 0.318681, 0.532164 });
        fourclassAnomalyData.add(new double[]{ 0.307692, -0.614035 });
        fourclassAnomalyData.add(new double[]{ 0.516484, 0.508772 });
        fourclassAnomalyData.add(new double[]{ 0.747253, -0.988304 });
        fourclassAnomalyData.add(new double[]{ 0.505495, -0.859649 });
        fourclassAnomalyData.add(new double[]{ 0.912088, -0.426901 });
        fourclassAnomalyData.add(new double[]{ 0.846154, -0.824561 });
        fourclassAnomalyData.add(new double[]{ 0.538462, 0.391813 });
        fourclassAnomalyData.add(new double[]{ 0.241758, 0.578947 });
        fourclassAnomalyData.add(new double[]{ 0.835165, -0.532164 });
        fourclassAnomalyData.add(new double[]{ 0.021978, -0.391813 });
        fourclassAnomalyData.add(new double[]{ 0.307692, 0.625731 });
        fourclassAnomalyData.add(new double[]{ -0.252747, -0.309942 });
        fourclassAnomalyData.add(new double[]{ 0.857143, -0.508772 });
        fourclassAnomalyData.add(new double[]{ 0.835165, -0.80117 });
        fourclassAnomalyData.add(new double[]{ -0.274725, -0.333333 });
        fourclassAnomalyData.add(new double[]{ -0.274725, -0.251462 });
        fourclassAnomalyData.add(new double[]{ -0.032967, 0.532164 });
        fourclassAnomalyData.add(new double[]{ 0.0769231, -0.298246 });
        fourclassAnomalyData.add(new double[]{ 0.725275, 0.415205 });
        fourclassAnomalyData.add(new double[]{ 0.835165, 0.321637 });
        fourclassAnomalyData.add(new double[]{ 0.78022, -0.660819 });
        fourclassAnomalyData.add(new double[]{ 0.362637, 0.532164 });
        fourclassAnomalyData.add(new double[]{ 0.307692, -0.520468 });
        fourclassAnomalyData.add(new double[]{ 0.868132, 0.169591 });
        fourclassAnomalyData.add(new double[]{ 0.824176, -0.122807 });
        fourclassAnomalyData.add(new double[]{ 0.021978, -0.380117 });
        fourclassAnomalyData.add(new double[]{ 0.758242, -0.146199 });
        fourclassAnomalyData.add(new double[]{ 0.494505, 0.637427 });
        fourclassAnomalyData.add(new double[]{ 0.494505, -0.74269 });
        fourclassAnomalyData.add(new double[]{ 0.681319, 0.321637 });
        fourclassAnomalyData.add(new double[]{ 0.428571, 0.649123 });
        fourclassAnomalyData.add(new double[]{ 0.153846, 0.672515 });
        fourclassAnomalyData.add(new double[]{ -0.032967, -0.321637 });
        fourclassAnomalyData.add(new double[]{ -0.120879, -0.356725 });
        fourclassAnomalyData.add(new double[]{ 0.692308, 0.333333 });
        fourclassAnomalyData.add(new double[]{ 0.153846, -0.321637 });
        fourclassAnomalyData.add(new double[]{ 0.857143, -0.754386 });
        fourclassAnomalyData.add(new double[]{ 0.769231, -0.555556 });
        fourclassAnomalyData.add(new double[]{ 0.703297, -0.789474 });
        fourclassAnomalyData.add(new double[]{ 0.857143, -0.812865 });
        fourclassAnomalyData.add(new double[]{ 0.0879121, -0.497076 });
        fourclassAnomalyData.add(new double[]{ 0.791209, -0.204678 });
        fourclassAnomalyData.add(new double[]{ 0.395604, -0.578947 });
        fourclassAnomalyData.add(new double[]{ 0.318681, -0.660819 });
        fourclassAnomalyData.add(new double[]{ -0.395604, -0.345029 });
        fourclassAnomalyData.add(new double[]{ 0.857143, -0.169591 });
        fourclassAnomalyData.add(new double[]{ 0.835165, -0.94152 });
        fourclassAnomalyData.add(new double[]{ 0.857143, -0.684211 });
        fourclassAnomalyData.add(new double[]{ 0.747253, 0.169591 });
        fourclassAnomalyData.add(new double[]{ 0.703297, -0.80117 });
        fourclassAnomalyData.add(new double[]{ 0.571429, -0.836257 });
        fourclassAnomalyData.add(new double[]{ 0.813187, -0.976608 });
        fourclassAnomalyData.add(new double[]{ 0.846154, 0.333333 });
        fourclassAnomalyData.add(new double[]{ 0.021978, -0.438596 });
        fourclassAnomalyData.add(new double[]{ 0.395604, -0.48538 });
        fourclassAnomalyData.add(new double[]{ 0.791209, 0.0994152 });
        fourclassAnomalyData.add(new double[]{ 0.857143, -0.22807 });
        fourclassAnomalyData.add(new double[]{ -0.153846, 0.719298 });
        fourclassAnomalyData.add(new double[]{ -0.10989, 0.625731 });
        fourclassAnomalyData.add(new double[]{ 0.252747, -0.48538 });
        fourclassAnomalyData.add(new double[]{ -0.428571, -0.309942 });
        fourclassAnomalyData.add(new double[]{ -0.0659341, -0.450292 });
        fourclassAnomalyData.add(new double[]{ 0.043956, 0.614035 });
        fourclassAnomalyData.add(new double[]{ 0.802198, 0.239766 });
        fourclassAnomalyData.add(new double[]{ -0.120879, 0.695906 });
        fourclassAnomalyData.add(new double[]{ 0.450549, -0.567251 });
        fourclassAnomalyData.add(new double[]{ 0.120879, 0.555556 });
        fourclassAnomalyData.add(new double[]{ 0.725275, 0.380117 });
        fourclassAnomalyData.add(new double[]{ 0.791209, -0.847953 });
        fourclassAnomalyData.add(new double[]{ 0.032967, -0.438596 });
        fourclassAnomalyData.add(new double[]{ -0.164835, 0.847953 });
        fourclassAnomalyData.add(new double[]{ 0.89011, 0.251462 });
        fourclassAnomalyData.add(new double[]{ -0.142857, -0.380117 });
        fourclassAnomalyData.add(new double[]{ 0.912088, -0.567251 });
        fourclassAnomalyData.add(new double[]{ 0.538462, -0.637427 });
        fourclassAnomalyData.add(new double[]{ 0.494505, -0.719298 });
        fourclassAnomalyData.add(new double[]{ 0.857143, -0.298246 });
        fourclassAnomalyData.add(new double[]{ 0.285714, -0.672515 });
        fourclassAnomalyData.add(new double[]{ -0.230769, -0.274854 });
        fourclassAnomalyData.add(new double[]{ 0.802198, -0.426901 });
        fourclassAnomalyData.add(new double[]{ 0.241758, 0.508772 });
        fourclassAnomalyData.add(new double[]{ 0.813187, -0.836257 });
        fourclassAnomalyData.add(new double[]{ 0.67033, 0.473684 });
        fourclassAnomalyData.add(new double[]{ 0.846154, 0.251462 });
        fourclassAnomalyData.add(new double[]{ 0.846154, -0.0760234 });
        fourclassAnomalyData.add(new double[]{ 0.516484, -0.637427 });
        fourclassAnomalyData.add(new double[]{ 0.791209, 0.450292 });
        fourclassAnomalyData.add(new double[]{ 0.197802, -0.461988 });
        fourclassAnomalyData.add(new double[]{ 0.406593, -0.508772 });
        fourclassAnomalyData.add(new double[]{ 0.274725, 0.520468 });
        fourclassAnomalyData.add(new double[]{ -0.0989011, 0.847953 });
        fourclassAnomalyData.add(new double[]{ 0.494505, -0.906433 });
        fourclassAnomalyData.add(new double[]{ 0.549451, 0.461988 });
        fourclassAnomalyData.add(new double[]{ 0.494505, 0.602339 });
        fourclassAnomalyData.add(new double[]{ 0.912088, -0.0994152 });
        fourclassAnomalyData.add(new double[]{ -0.252747, -0.450292 });
        fourclassAnomalyData.add(new double[]{ -0.010989, 0.766082 });
        fourclassAnomalyData.add(new double[]{ 0.846154, -0.953216 });
        fourclassAnomalyData.add(new double[]{ 0.142857, -0.345029 });
        fourclassAnomalyData.add(new double[]{ -0.252747, -0.333333 });
        fourclassAnomalyData.add(new double[]{ 0.241758, -0.415205 });
        fourclassAnomalyData.add(new double[]{ 0.351648, -0.461988 });
        fourclassAnomalyData.add(new double[]{ 0.0769231, -0.274854 });
        fourclassAnomalyData.add(new double[]{ -0.010989, 0.578947 });
        fourclassAnomalyData.add(new double[]{ 0.571429, -0.74269 });
        fourclassAnomalyData.add(new double[]{ 0.901099, -0.0643275 });
        fourclassAnomalyData.add(new double[]{ 0.230769, -0.590643 });
        fourclassAnomalyData.add(new double[]{ 0.923077, -0.695906 });
        fourclassAnomalyData.add(new double[]{ -0.142857, 0.789474 });
        fourclassAnomalyData.add(new double[]{ 0.791209, -0.988304 });
        fourclassAnomalyData.add(new double[]{ 0.43956, 0.567251 });
        fourclassAnomalyData.add(new double[]{ 0.725275, 0.204678 });
        fourclassAnomalyData.add(new double[]{ -0.142857, 0.625731 });
        fourclassAnomalyData.add(new double[]{ 0.868132, 0.309942 });
        fourclassAnomalyData.add(new double[]{ 0.428571, -0.614035 });
        fourclassAnomalyData.add(new double[]{ 0.626374, 0.578947 });
        fourclassAnomalyData.add(new double[]{ 0.483516, -0.894737 });
        fourclassAnomalyData.add(new double[]{ 0.857143, -0.134503 });
        fourclassAnomalyData.add(new double[]{ 0.274725, -0.461988 });
        fourclassAnomalyData.add(new double[]{ -0.472527, -0.263158 });
        fourclassAnomalyData.add(new double[]{ 0.0769231, 0.707602 });
        fourclassAnomalyData.add(new double[]{ 0.879121, -0.263158 });
        fourclassAnomalyData.add(new double[]{ 0.769231, -1.0 });
        fourclassAnomalyData.add(new double[]{ -0.197802, -0.298246 });
        fourclassAnomalyData.add(new double[]{ -0.263736, -0.274854 });
        fourclassAnomalyData.add(new double[]{ 0.89011, -0.0409357 });
        fourclassAnomalyData.add(new double[]{ -0.307692, -0.356725 });
        fourclassAnomalyData.add(new double[]{ 0.714286, 0.263158 });
        fourclassAnomalyData.add(new double[]{ 0.637363, 0.391813 });
        fourclassAnomalyData.add(new double[]{ 0.351648, 0.508772 });
        fourclassAnomalyData.add(new double[]{ -0.10989, -0.368421 });
        fourclassAnomalyData.add(new double[]{ 0.758242, -0.672515 });
        fourclassAnomalyData.add(new double[]{ -0.406593, -0.356725 });
        fourclassAnomalyData.add(new double[]{ -0.164835, -0.28655 });
        fourclassAnomalyData.add(new double[]{ 0.835165, -0.590643 });
        fourclassAnomalyData.add(new double[]{ 0.516484, -0.894737 });
        fourclassAnomalyData.add(new double[]{ 0.89011, 0.146199 });
        fourclassAnomalyData.add(new double[]{ -0.032967, 0.707602 });
        fourclassAnomalyData.add(new double[]{ 0.461538, -0.871345 });
        fourclassAnomalyData.add(new double[]{ 0.417582, -0.707602 });
        fourclassAnomalyData.add(new double[]{ 0.142857, 0.684211 });
        fourclassAnomalyData.add(new double[]{ 0.637363, -0.929825 });
        fourclassAnomalyData.add(new double[]{ 0.021978, -0.426901 });
        fourclassAnomalyData.add(new double[]{ 0.78022, -0.74269 });
        fourclassAnomalyData.add(new double[]{ 0.791209, -0.929825 });
        fourclassAnomalyData.add(new double[]{ 0.0769231, -0.403509 });
        fourclassAnomalyData.add(new double[]{ 0.67033, -0.847953 });
        fourclassAnomalyData.add(new double[]{ 0.241758, 0.555556 });
        fourclassAnomalyData.add(new double[]{ 0.010989, 0.637427 });
        fourclassAnomalyData.add(new double[]{ 0.032967, -0.28655 });
        fourclassAnomalyData.add(new double[]{ 0.0989011, 0.508772 });
        fourclassAnomalyData.add(new double[]{ -0.230769, -0.426901 });
        fourclassAnomalyData.add(new double[]{ 0.846154, -0.321637 });
        fourclassAnomalyData.add(new double[]{ 0.626374, -0.953216 });
        fourclassAnomalyData.add(new double[]{ 0.89011, -0.0760234 });
        fourclassAnomalyData.add(new double[]{ 0.846154, -0.22807 });
        fourclassAnomalyData.add(new double[]{ -0.010989, -0.356725 });
        fourclassAnomalyData.add(new double[]{ -0.043956, 0.637427 });
        fourclassAnomalyData.add(new double[]{ -0.043956, -0.368421 });
        fourclassAnomalyData.add(new double[]{ -0.417582, -0.356725 });
        fourclassAnomalyData.add(new double[]{ 0.912088, -0.192982 });
        fourclassAnomalyData.add(new double[]{ -0.395604, -0.380117 });
        fourclassAnomalyData.add(new double[]{ 0.175824, -0.356725 });
        fourclassAnomalyData.add(new double[]{ 0.725275, -0.847953 });
        fourclassAnomalyData.add(new double[]{ 0.857143, -0.28655 });
        fourclassAnomalyData.add(new double[]{ 0.318681, -0.54386 });
        fourclassAnomalyData.add(new double[]{ 0.813187, -0.637427 });
        fourclassAnomalyData.add(new double[]{ -0.0989011, 0.859649 });
        fourclassAnomalyData.add(new double[]{ -0.274725, -0.321637 });
        fourclassAnomalyData.add(new double[]{ 0.791209, -0.380117 });
        fourclassAnomalyData.add(new double[]{ 0.769231, 0.461988 });
        fourclassAnomalyData.add(new double[]{ 0.021978, -0.274854 });
        fourclassAnomalyData.add(new double[]{ 0.571429, -0.777778 });
        fourclassAnomalyData.add(new double[]{ 0.307692, -0.578947 });
        fourclassAnomalyData.add(new double[]{ 0.461538, 0.450292 });
        fourclassAnomalyData.add(new double[]{ -0.153846, -0.426901 });
        fourclassAnomalyData.add(new double[]{ 0.912088, -0.0175439 });
        fourclassAnomalyData.add(new double[]{ 0.0549451, -0.368421 });
        fourclassAnomalyData.add(new double[]{ 0.78022, -0.695906 });
        fourclassAnomalyData.add(new double[]{ 0.89011, -0.380117 });
        fourclassAnomalyData.add(new double[]{ -0.43956, -0.391813 });
        fourclassAnomalyData.add(new double[]{ 0.241758, 0.532164 });
        fourclassAnomalyData.add(new double[]{ 0.252747, -0.497076 });
        fourclassAnomalyData.add(new double[]{ -0.0989011, -0.461988 });
        fourclassAnomalyData.add(new double[]{ 0.351648, -0.637427 });
        fourclassAnomalyData.add(new double[]{ 0.538462, -0.859649 });
        fourclassAnomalyData.add(new double[]{ 0.494505, -0.625731 });
        fourclassAnomalyData.add(new double[]{ 0.78022, -0.28655 });
        fourclassAnomalyData.add(new double[]{ -0.307692, -0.333333 });
        fourclassAnomalyData.add(new double[]{ -0.241758, -0.28655 });
        fourclassAnomalyData.add(new double[]{ 0.472527, -0.578947 });
        fourclassAnomalyData.add(new double[]{ -0.208791, -0.415205 });
        fourclassAnomalyData.add(new double[]{ -0.0769231, 0.789474 });
        fourclassAnomalyData.add(new double[]{ -0.0549451, 0.80117 });
        fourclassAnomalyData.add(new double[]{ 0.868132, -0.637427 });
        fourclassAnomalyData.add(new double[]{ 0.868132, 0.251462 });
        fourclassAnomalyData.add(new double[]{ -0.0549451, 0.567251 });
        fourclassAnomalyData.add(new double[]{ 0.043956, 0.730994 });
        fourclassAnomalyData.add(new double[]{ 0.835165, -0.0760234 });

        int trainingDataSize = 500;
        RandomCutForest forest = RandomCutForest.builder()
                .numberOfTrees(100)
                .sampleSize(trainingDataSize)
                .outputAfter(trainingDataSize)
                .dimensions(2) // still required!
                .randomSeed(123)
                .storeSequenceIndexesEnabled(true)
                .centerOfMassEnabled(true)
                .build();

        double max = Double.MIN_VALUE;
        double min = Double.MAX_VALUE;
        int i = 0;
        for (double[] point : fourclassData) {
            i ++;
            double score = forest.getAnomalyScore(point);
            if (score < min) {
                min = score;
            }
            if (score > max) {
                max = score;
            }
            if (i <= trainingDataSize) {
                forest.update(point);
            }
            System.out.println(i + ": Anomaly Score: " + score);
        }
        System.out.println("1==================== min: " + min + ", max: " + max);

        max = Double.MIN_VALUE;
        min = Double.MAX_VALUE;
        int anomalyCount = 0;
        int j = 0;
        for (double[] point : fourclassAnomalyData) {
            j ++;
            double score = forest.getAnomalyScore(point);
            if (score < min) {
                min = score;
            }
            if (score > max) {
                max = score;
            }
            if (score > 1.0) {
                anomalyCount++;
            }
            //System.out.println(j + " ---- : Anomaly Score: " + score);
            System.out.println(String.format("{ \"index\" : { \"_index\" : \"test_data_ad_predict_result_rcf2\", \"_id\" : \"%s\" } }", j));
            System.out.println(String.format("{\"A\":%s,\"B\":%s,\"score\":%s,\"anomaly_type\":\"%s\"}",
                    point[0], point[1], score, score > 1? "ANOMALOUS" : "EXPECTED"));
        }
//        System.out.println("2==================== min: " + min + ", max: " + max + ", anomalyCount: " + anomalyCount);

        max = Double.MIN_VALUE;
        min = Double.MAX_VALUE;
        anomalyCount = 0;
        int m = 0;
        for (double[] point : fourclassData) {
            m++;
            double score = forest.getAnomalyScore(point);
            if (score < min) {
                min = score;
            }
            if (score > max) {
                max = score;
            }
            if (score > 1.0) {
                anomalyCount++;
            }
            //System.out.println(m + " +++++ : Anomaly Score: " + score);
            System.out.println(String.format("{ \"index\" : { \"_index\" : \"test_data_ad_predict_result_rcf2\", \"_id\" : \"%s\" } }", j + m));
            System.out.println(String.format("{\"A\":%s,\"B\":%s,\"score\":%s,\"anomaly_type\":\"%s\"}",
                    point[0], point[1], score, score > 1? "ANOMALOUS" : "EXPECTED"));
        }
//        System.out.println("3==================== min: " + min + ", max: " + max + ", anomalyCount: " + anomalyCount);
    }

    private Model trainKMeansModel() {
        KMeansParams parameters = KMeansParams.builder()
                .centroids(2)
                .iterations(10)
                .distanceType(KMeansParams.DistanceType.EUCLIDEAN)
                .build();
        DataFrame trainDataFrame = constructKMeansDataFrame(100);
        MLInputDataset inputDataset = DataFrameInputDataset.builder().dataFrame(trainDataFrame).build();
        Input mlInput = MLInput.builder().algorithm(FunctionName.KMEANS).parameters(parameters).inputDataset(inputDataset).build();
        return MLEngine.train(mlInput);
    }

    private Model trainLinearRegressionModel() {
        LinearRegressionParams parameters = LinearRegressionParams.builder()
                .objectiveType(LinearRegressionParams.ObjectiveType.SQUARED_LOSS)
                .optimizerType(LinearRegressionParams.OptimizerType.ADAM)
                .learningRate(0.01)
                .epsilon(1e-6)
                .beta1(0.9)
                .beta2(0.99)
                .target("price")
                .build();
        DataFrame trainDataFrame = constructLinearRegressionTrainDataFrame();
        MLInputDataset inputDataset = DataFrameInputDataset.builder().dataFrame(trainDataFrame).build();
        Input mlInput = MLInput.builder().algorithm(FunctionName.LINEAR_REGRESSION).parameters(parameters).inputDataset(inputDataset).build();

        return MLEngine.train(mlInput);
    }
}