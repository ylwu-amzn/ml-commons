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

package org.opensearch.ml.engine.algorithms.ad;

import lombok.extern.log4j.Log4j2;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataframe.DataFrameBuilder;
import org.opensearch.ml.common.parameter.FunctionName;
import org.opensearch.ml.common.parameter.AnomalyDetectionParams;
import org.opensearch.ml.common.parameter.MLAlgoParams;
import org.opensearch.ml.common.parameter.MLOutput;
import org.opensearch.ml.common.parameter.MLPredictionOutput;
import org.opensearch.ml.engine.Model;
import org.opensearch.ml.engine.Predictable;
import org.opensearch.ml.engine.Trainable;
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
import java.util.Optional;

/**
 * Wrap Tribuo's anomaly detection based on one-class SVM (libSVM).
 *
 */
@Log4j2
@Function(FunctionName.ANOMALY_DETECTION)
public class AnomalyDetection implements Trainable, Predictable {
    public static final int VERSION = 1;
//    private static double DEFAULT_GAMMA = 1.0;
//    private static double DEFAULT_NU = 0.1;
    private static KernelType DEFAULT_KERNEL_TYPE = KernelType.RBF;

    private AnomalyDetectionParams parameters;

    public AnomalyDetection() {}

    public AnomalyDetection(MLAlgoParams parameters) {
        this.parameters = parameters == null ? AnomalyDetectionParams.builder().build() : (AnomalyDetectionParams)parameters;
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
            result.put("anomaly_type", e.getOutput().getType().name());
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
        KernelType kernelType = parseKernelType();
        log.info("---------------- kernel type " + kernelType);
        SVMParameters params = new SVMParameters<>(new SVMAnomalyType(SVMAnomalyType.SVMMode.ONE_CLASS), kernelType);
//        Double gamma = Optional.ofNullable(parameters.getGamma()).orElse(DEFAULT_GAMMA);
//        Double nu = Optional.ofNullable(parameters.getNu()).orElse(DEFAULT_NU);
        if (parameters.getGamma() != null) {
            params.setGamma(parameters.getGamma());
        }
        if (parameters.getNu() != null) {
            params.setNu(parameters.getNu());
        }
        if (parameters.getCost() != null) {
            params.setCost(parameters.getCost());
        }
        if (parameters.getCoeff() != null) {
            params.setCoeff(parameters.getCoeff());
        }
        if (parameters.getEpsilon() != null) {
            params.setEpsilon(parameters.getEpsilon());
        }

        if (parameters.getDegree() != null) {
            params.setDegree(parameters.getDegree());
        }
//        params.setCacheSize();
        //params.setProbability();
        MutableDataset<Event> data = TribuoUtil.generateDataset(dataFrame, new AnomalyFactory(),
                "Anomaly detection LibSVM training data from OpenSearch", TribuoOutputType.ANOMALY_DETECTION_LIBSVM);

        LibSVMAnomalyTrainer trainer = new LibSVMAnomalyTrainer(params);

        LibSVMModel libSVMModel = trainer.train(data);
        ((LibSVMAnomalyModel)libSVMModel).getNumberOfSupportVectors();
        Model model = new Model();
        model.setName(FunctionName.ANOMALY_DETECTION.name());
        model.setVersion(VERSION);
        model.setContent(ModelSerDeSer.serialize(libSVMModel));
        return model;
    }

    private KernelType parseKernelType() {
        KernelType kernelType = DEFAULT_KERNEL_TYPE;
        if (parameters.getKernelType() == null) {
            return kernelType;
        }
        switch (parameters.getKernelType()){
            case LINEAR:
                kernelType = KernelType.LINEAR;
                break;
            case POLY:
                kernelType = KernelType.POLY;
                break;
            case RBF:
                kernelType = KernelType.RBF;
                break;
            case SIGMOID:
                kernelType = KernelType.SIGMOID;
                break;
            default:
                break;
        }
        return kernelType;
    }
}
