/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.rcf;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.state.RandomCutForestMapper;
import com.amazon.randomcutforest.state.RandomCutForestState;
import io.protostuff.Schema;
import io.protostuff.runtime.RuntimeSchema;
import lombok.extern.log4j.Log4j2;
import org.opensearch.ml.common.dataframe.ColumnMeta;
import org.opensearch.ml.common.dataframe.ColumnValue;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataframe.DataFrameBuilder;
import org.opensearch.ml.common.dataframe.Row;
import org.opensearch.ml.common.input.parameter.ad.BatchRCFParams;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.input.parameter.MLAlgoParams;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.common.output.MLPredictionOutput;
import org.opensearch.ml.common.Model;
import org.opensearch.ml.engine.TrainAndPredictable;
import org.opensearch.ml.engine.annotation.Function;
import org.opensearch.ml.engine.utils.ModelSerDeSer;

import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Use RCF to detect non-time-series data.
 */
@Log4j2
@Function(FunctionName.BATCH_RCF)
public class BatchRandomCutForest implements TrainAndPredictable {
    private static final int DEFAULT_NUMBER_OF_TREES = 30;
    private static final int DEFAULT_OUTPUT_AFTER = 32;
    private static final int DEFAULT_SAMPLES_SIZE = 256; // how many nodes per tree
    private static final double DEFAULT_ANOMALY_SCORE_THRESHOLD = 1.0;

    private Integer numberOfTrees = DEFAULT_NUMBER_OF_TREES;
    private Integer sampleSize = DEFAULT_SAMPLES_SIZE;
    private Integer outputAfter = DEFAULT_OUTPUT_AFTER;
    private Double anomalyScoreThreshold = DEFAULT_ANOMALY_SCORE_THRESHOLD;
    private Integer trainingDataSize;

    private static final RandomCutForestMapper rcfMapper = new RandomCutForestMapper();

    public BatchRandomCutForest(){}

    public BatchRandomCutForest(MLAlgoParams parameters) {
        rcfMapper.setSaveExecutorContextEnabled(true);
        if (parameters != null) {
            BatchRCFParams rcfParams = (BatchRCFParams) parameters;
            this.numberOfTrees = Optional.ofNullable(rcfParams.getNumberOfTrees()).orElse(DEFAULT_NUMBER_OF_TREES);
            this.sampleSize = Optional.ofNullable(rcfParams.getSampleSize()).orElse(DEFAULT_SAMPLES_SIZE);
            this.outputAfter = Optional.ofNullable(rcfParams.getOutputAfter()).orElse(DEFAULT_OUTPUT_AFTER);
            this.anomalyScoreThreshold = Optional.ofNullable(rcfParams.getAnomalyScoreThreshold()).orElse(DEFAULT_ANOMALY_SCORE_THRESHOLD);
            this.trainingDataSize = rcfParams.getTrainingDataSize();
        }
    }

    @Override
    public MLOutput predict(DataFrame dataFrame, Model model) {
        if (model == null) {
            throw new IllegalArgumentException("No model found for batch RCF prediction.");
        }
        RandomCutForestState state = RCFModelSerDeSer.deserializeRCF(model.getContent());
        RandomCutForest forest = rcfMapper.toModel(state);
        List<Map<String, Object>> predictResult = process(dataFrame, forest, 0);
        return MLPredictionOutput.builder().predictionResult(DataFrameBuilder.load(predictResult)).build();
    }

    @Override
    public Model train(DataFrame dataFrame) {
        RandomCutForest forest = createRandomCutForest(dataFrame);
        Integer actualTrainingDataSize = trainingDataSize == null ? dataFrame.size() : trainingDataSize;
        process(dataFrame, forest, actualTrainingDataSize);
        Model model = new Model();
        model.setName(FunctionName.BATCH_RCF.name());
        model.setVersion(1);
        RandomCutForestState state = rcfMapper.toState(forest);
        model.setContent(RCFModelSerDeSer.serializeRCF(state));
        return model;
    }

    @Override
    public MLOutput trainAndPredict(DataFrame dataFrame) {
        RandomCutForest forest = createRandomCutForest(dataFrame);
        Integer actualTrainingDataSize = trainingDataSize == null ? dataFrame.size() : trainingDataSize;
        List<Map<String, Object>> predictResult = process(dataFrame, forest, actualTrainingDataSize);
        return MLPredictionOutput.builder().predictionResult(DataFrameBuilder.load(predictResult)).build();
    }

    private List<Map<String, Object>> process(DataFrame dataFrame, RandomCutForest forest, Integer actualTrainingDataSize) {
        List<Double> pointList = new ArrayList<>();
        ColumnMeta[] columnMetas = dataFrame.columnMetas();
        List<Map<String, Object>> predictResult = new ArrayList<>();

        for (int rowNum = 0; rowNum< dataFrame.size(); rowNum++) {
            for (int i = 0; i < columnMetas.length; i++) {
                Row row = dataFrame.getRow(rowNum);
                ColumnValue value = row.getValue(i);
                pointList.add(value.doubleValue());
            }
            double[] point = pointList.stream().mapToDouble(d -> d).toArray();
            pointList.clear();
            double anomalyScore = forest.getAnomalyScore(point);
            if (actualTrainingDataSize == null || rowNum < actualTrainingDataSize) {
                forest.update(point);
            }
            Map<String, Object> result = new HashMap<>();
            result.put("score", anomalyScore);
            result.put("anomalous", anomalyScore > anomalyScoreThreshold);
            predictResult.add(result);
        }
        return predictResult;
    }

    private RandomCutForest createRandomCutForest(DataFrame dataFrame) {
        //TODO: add memory estimation of RCF. Will be better if support memory estimation in RCF
        RandomCutForest forest = RandomCutForest
                .builder()
                .dimensions(dataFrame.columnMetas().length)
                .numberOfTrees(numberOfTrees)
                .sampleSize(sampleSize)
                .outputAfter(outputAfter)
                .parallelExecutionEnabled(false)
                .build();
        return forest;
    }

}
