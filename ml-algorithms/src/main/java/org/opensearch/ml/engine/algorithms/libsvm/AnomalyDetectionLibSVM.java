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

import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataframe.DataFrameBuilder;
import org.opensearch.ml.common.parameter.FunctionName;
import org.opensearch.ml.common.parameter.AnomalyDetectionLibSVMParams;
import org.opensearch.ml.common.parameter.MLAlgoParams;
import org.opensearch.ml.common.parameter.MLOutput;
import org.opensearch.ml.common.parameter.MLPredictionOutput;
import org.opensearch.ml.engine.MLAlgo;
import org.opensearch.ml.engine.MLAlgoMetaData;
import org.opensearch.ml.engine.Model;
import org.opensearch.ml.engine.annotation.Function;
import org.opensearch.ml.engine.contants.TribuoOutputType;
import org.opensearch.ml.engine.utils.ModelSerDeSer;
import org.opensearch.ml.engine.utils.TribuoUtil;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.anomaly.AnomalyFactory;
import org.tribuo.anomaly.Event;
import org.tribuo.anomaly.libsvm.LibSVMAnomalyModel;
import org.tribuo.anomaly.libsvm.LibSVMAnomalyTrainer;
import org.tribuo.anomaly.libsvm.SVMAnomalyType;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.SVMParameters;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Wrap Tribuo's anomaly detection based on one-class SVM (libSVM).
 *
 */
@Function(FunctionName.ANOMALY_DETECTION_LIBSVM)
public class AnomalyDetectionLibSVM implements MLAlgo {
    private static double DEFAULT_GAMMA = 1.0;
    private static double DEFAULT_NU = 0.1;

    private AnomalyDetectionLibSVMParams parameters;

    public AnomalyDetectionLibSVM() {}

    public AnomalyDetectionLibSVM(MLAlgoParams parameters) {
        this.parameters = parameters == null ? AnomalyDetectionLibSVMParams.builder().build() : (AnomalyDetectionLibSVMParams)parameters;
        validateParameters();
    }

    private void validateParameters() {

        if (parameters.getGamma() != null && parameters.getGamma() <= 0) {
            throw new IllegalArgumentException("gamma should be positive.");
        }

        if (parameters.getNu() != null && parameters.getNu() <= 0) {
            throw new IllegalArgumentException("nu should be positive.");
        }

    }

    @Override
    public MLOutput predict(DataFrame dataFrame, Model model) {
        if (model == null) {
            throw new IllegalArgumentException("No model found for KMeans prediction.");
        }

        List<Prediction<Event>> predictions;
        MutableDataset<Event> predictionDataset = TribuoUtil.generateDataset(dataFrame, new AnomalyFactory(),
                "Anomaly detection LibSVM prediction data from OpenSearch", TribuoOutputType.ANOMALY_DETECTION_LIBSVM);
        LibSVMModel libSVMAnomalyModel = (LibSVMModel) ModelSerDeSer.deserialize(model.getContent());
        predictions = libSVMAnomalyModel.predict(predictionDataset);

        List<Map<String, Object>> adResults = new ArrayList<>();
        predictions.forEach(e -> {
            Map<String, Object> result = new HashMap<>();
            result.put("event_type", e.getOutput().getType().name());
            result.put("score", e.getOutput().getScore());
            Example<Event> example = e.getExample();
            for (Feature feature : example) {
                result.put(feature.getName(), feature.getValue());
            }
            adResults.add(result);
        });

        return MLPredictionOutput.builder().predictionResult(DataFrameBuilder.load(adResults)).build();
    }

    @Override
    public Model train(DataFrame dataFrame) {
        SVMParameters params = new SVMParameters<>(new SVMAnomalyType(SVMAnomalyType.SVMMode.ONE_CLASS), KernelType.RBF);
        params.setGamma(1.0);
        params.setNu(0.1);
        MutableDataset<Event> data = TribuoUtil.generateDataset(dataFrame, new AnomalyFactory(),
                "Anomaly detection LibSVM training data from OpenSearch", TribuoOutputType.ANOMALY_DETECTION_LIBSVM);

        LibSVMAnomalyTrainer trainer = new LibSVMAnomalyTrainer(params);

        LibSVMModel libSVMModel = trainer.train(data);
        ((LibSVMAnomalyModel)libSVMModel).getNumberOfSupportVectors();
        Model model = new Model();
        model.setName(FunctionName.ANOMALY_DETECTION_LIBSVM.name());
        model.setVersion(1);
        model.setContent(ModelSerDeSer.serialize(libSVMModel));
        return model;
    }

    @Override
    public MLAlgoMetaData getMetaData() {
        return MLAlgoMetaData.builder().name(FunctionName.ANOMALY_DETECTION_LIBSVM.name())
                .description("Anomaly detection based on one-class SVM.")
                .version("1.0")
                .predictable(true)
                .trainable(true)
                .executable(false)
                .build();
    }
}
