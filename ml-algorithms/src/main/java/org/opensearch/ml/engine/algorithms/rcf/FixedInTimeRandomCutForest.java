/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.rcf;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import lombok.extern.log4j.Log4j2;
import org.opensearch.ml.common.dataframe.ColumnMeta;
import org.opensearch.ml.common.dataframe.ColumnType;
import org.opensearch.ml.common.dataframe.ColumnValue;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataframe.DataFrameBuilder;
import org.opensearch.ml.common.dataframe.Row;
import org.opensearch.ml.common.exception.MLValidationException;
import org.opensearch.ml.common.parameter.FunctionName;
import org.opensearch.ml.common.parameter.MLAlgoParams;
import org.opensearch.ml.common.parameter.MLOutput;
import org.opensearch.ml.common.parameter.MLPredictionOutput;
import org.opensearch.ml.common.parameter.Model;
import org.opensearch.ml.common.parameter.RCFParams;
import org.opensearch.ml.common.parameter.SampleAlgoOutput;
import org.opensearch.ml.engine.TrainAndPredictable;
import org.opensearch.ml.engine.annotation.Function;
import org.opensearch.ml.engine.utils.ModelSerDeSer;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.TimeZone;
import java.util.concurrent.atomic.AtomicReference;


@Log4j2
@Function(FunctionName.RCF)
public class FixedInTimeRandomCutForest implements TrainAndPredictable {
    private static final int DEFAULT_SHINGLE_SIZE = 8;
    private static final String DEFAULT_TIME_ZONE = "UTC";
    private int shingleSize;
    private Integer trainingDataSize;
    private String timeField;
    private String dateFormat;
    private String timeZone;
    private float anomalyThresholdRatio;
    private DateFormat simpleDateFormat;

    private final int NUM_MIN_SAMPLES = 32;
    private final int NUM_SAMPLES_PER_TREE = 256;
    private final int NUM_TREES = 30;
    private final double TIME_DECAY = 0.0001;
    private final double THRESHOLD_MIN_PVALUE = 0.995;

    public FixedInTimeRandomCutForest(){}

    public FixedInTimeRandomCutForest(MLAlgoParams parameters) {
        RCFParams rcfParams = (RCFParams) parameters;
        this.shingleSize = Optional.ofNullable(rcfParams.getShingleSize()).orElse(DEFAULT_SHINGLE_SIZE);
        this.timeField = rcfParams.getTimeField();
        this.dateFormat = rcfParams.getDateFormat();
        this.timeZone = Optional.ofNullable(rcfParams.getTimeZone()).orElse(DEFAULT_TIME_ZONE);
        if (timeField != null && dateFormat != null) {
            simpleDateFormat = new SimpleDateFormat(dateFormat);
            simpleDateFormat.setTimeZone(TimeZone.getTimeZone(timeZone));
        }
        this.trainingDataSize = rcfParams.getTrainingDataSize();
    }

    @Override
    public MLOutput predict(DataFrame dataFrame, Model model) {
        if (model == null) {
            throw new IllegalArgumentException("No model found for KMeans prediction.");
        }
        AtomicReference<Double> sum = new AtomicReference<>((double) 0);
        dataFrame.forEach(row -> {
            row.forEach(item -> sum.updateAndGet(v -> v + item.doubleValue()));
        });
        return SampleAlgoOutput.builder().sampleResult(sum.get()).build();
    }

    @Override
    public Model train(DataFrame dataFrame) {
        ThresholdedRandomCutForest forest = createThresholdedRandomCutForest(dataFrame);

        Model model = new Model();
        model.setName(FunctionName.RCF.name());
        model.setVersion(1);
        model.setContent(ModelSerDeSer.serialize(forest));
        return model;
    }

    private ThresholdedRandomCutForest createThresholdedRandomCutForest(DataFrame dataFrame) {
        //TODO: add memory estimation of RCF. Will be better if support memory estimation in RCF
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder()
                .dimensions(shingleSize * (dataFrame.columnMetas().length - 1))
                .sampleSize(NUM_SAMPLES_PER_TREE)
                .numberOfTrees(NUM_TREES)
                .timeDecay(TIME_DECAY)
                .outputAfter(NUM_MIN_SAMPLES)
                .initialAcceptFraction(NUM_MIN_SAMPLES * 1.0d / NUM_SAMPLES_PER_TREE)
                .parallelExecutionEnabled(false)
                .compact(true)
                .precision(Precision.FLOAT_32)
                .boundingBoxCacheFraction(0)
                //.boundingBoxCacheFraction(1)
                .shingleSize(shingleSize)
                .internalShinglingEnabled(true)
                .anomalyRate(1 - THRESHOLD_MIN_PVALUE)
                .build();
        return forest;
    }

    @Override
    public MLOutput trainAndPredict(DataFrame dataFrame) {
        ThresholdedRandomCutForest forest = null;
        RandomCutForest rcfModel = null;
        List<Double> pointList = new ArrayList<>();
        ColumnMeta[] columnMetas = dataFrame.columnMetas();
        List<Map<String, Object>> predictResult = new ArrayList<>();
        if (timeField == null) {
            rcfModel = RandomCutForest
                    .builder()
                    .dimensions(dataFrame.columnMetas().length)
                    .numberOfTrees(NUM_TREES)
                    .timeDecay(TIME_DECAY)
                    .sampleSize(NUM_SAMPLES_PER_TREE)
                    .outputAfter(NUM_MIN_SAMPLES)
                    .parallelExecutionEnabled(false)
                    .build();
        } else {
            forest = createThresholdedRandomCutForest(dataFrame);
        }
        for (int rowNum = 0; rowNum<dataFrame.size(); rowNum++) {
            Row row = dataFrame.getRow(rowNum);
            long timestamp = -1;
            for (int i = 0; i < columnMetas.length; i++) {
                ColumnMeta columnMeta = columnMetas[i];
                ColumnValue value = row.getValue(i);

                //TODo: if user use query input data set, parse time field in query result to long
                if (timeField != null && timeField.equals(columnMeta.getName())) { // TODO: sort dataframe by time field with asc order
                    ColumnType columnType = columnMeta.getColumnType();
                    if (columnType == ColumnType.LONG ) {
                        timestamp = value.longValue();
                    } else if (columnType == ColumnType.STRING) {
                        try {
                            timestamp = simpleDateFormat.parse(value.stringValue()).getTime();
                        } catch (ParseException e) {
                            log.error("Failed to parse timestamp " + value.stringValue(), e);
                            throw new MLValidationException("Failed to parse timestamp " + value.stringValue());
                        }
                    } else  {
                        throw new MLValidationException("Wrong data type of time field. Should use LONG or STRING, but got " + columnType);
                    }
                } else {
                    pointList.add(value.doubleValue());
                }
            }
            double[] point = pointList.stream().mapToDouble(d -> d).toArray();
            pointList.clear();
            Map<String, Object> result = new HashMap<>();

            if (timeField != null) {
                AnomalyDescriptor process = forest.process(point, timestamp);
                result.put(timeField, timestamp);
                result.put("score", process.getRCFScore());
                result.put("anomaly_grade", process.getAnomalyGrade());
            } else {
                //If timefield is null, just use RCF without shingle
                double anomalyScore = rcfModel.getAnomalyScore(point);
                if (this.trainingDataSize == null || rowNum < this.trainingDataSize) {
                    rcfModel.update(point);
                }
                result.put("score", anomalyScore);// calculate is anomaly or not based on threshold ratio
            }
            predictResult.add(result);
        }
        return MLPredictionOutput.builder().predictionResult(DataFrameBuilder.load(predictResult)).build();
    }


}
