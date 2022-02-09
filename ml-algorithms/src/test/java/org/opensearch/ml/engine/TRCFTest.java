/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.smelter.base.DataFrame;
import com.smelter.base.SortMethod;
import com.smelter.function.map.MapFunction;
import gnu.trove.impl.sync.TSynchronizedIntByteMap;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import java.io.IOException;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TimeZone;

import static com.smelter.utils.DateUtil.YEAR_MONTH_DAY_HOUR_MIN_SECOND_FORMAT;

public class TRCFTest {
    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();


    public List<double[]> createTrainingDataForMachineFailure() throws ParseException {
        List<double[]> data = new ArrayList<>();
        DataFrame df = new DataFrame("/Users/ylwu/code/ml/NAB/data/realKnownCause/machine_temperature_system_failure.csv");
        df.asType("timestamp", Date.class);
        df.asType("value", Double.class);
//        System.out.println(df.first(20));
        DataFrame loc = df.loc(df.where("timestamp>=\"2013-12-18 01:00:00\" and timestamp<=\"2013-12-31 01:00:00\""));
//        System.out.println(loc.first(20));
//        System.out.println(loc.getSize());
        List<Double> value = loc.getColumnData("value");
//        System.out.println(Arrays.toString(value.toArray()));
        for (Double v : value) {
            data.add(new double[]{v});
        }
        return data;
    }

    public DataFrame createTrainingDataForMachineFailure2() throws ParseException {
        List<double[]> data = new ArrayList<>();
        DataFrame df = new DataFrame("/Users/ylwu/code/ml/NAB/data/realKnownCause/machine_temperature_system_failure.csv");
        df.asType("timestamp", Date.class);
        df.asType("value", Double.class);
//        System.out.println(df.first(20));
        DataFrame loc = df.loc(df.where("timestamp>=\"2013-12-18 01:00:00\" and timestamp<=\"2013-12-31 01:00:00\""));
//        System.out.println(loc.first(20));
//        System.out.println(loc.getSize());
        List<Double> value = loc.getColumnData("value");
//        System.out.println(Arrays.toString(value.toArray()));
        for (Double v : value) {
            data.add(new double[]{v});
        }
        return loc;
    }

    public DataFrame createAllDataForMachineFailure() throws ParseException {
        List<double[]> data = new ArrayList<>();
        DataFrame df = new DataFrame("/Users/ylwu/code/ml/NAB/data/realKnownCause/machine_temperature_system_failure.csv");
        df.asType("timestamp", Date.class);
        df.asType("value", Double.class);
        List<Double> value = df.getColumnData("value");
//        System.out.println(Arrays.toString(value.toArray()));
        for (Double v : value) {
            data.add(new double[]{v});
        }
        return df;
    }


    public List<double[]> createTrainingDataForNycTaxi(int shingleSize, int trainingDataSize) throws ParseException {
        List<double[]> data = new ArrayList<>();
        DataFrame df = new DataFrame("/local/home/ylwu/code/os/ml/NAB/data/realKnownCause/nyc_taxi.csv");
        df.asType("timestamp", Date.class);
        df.asType("value", Double.class);
        List<Double> shingle = new ArrayList<>();
        for (int i = 0; i < trainingDataSize; i++) {
            for (int j = i; j < i + shingleSize; j++) {
                shingle.add((Double) df.getRowData(j)[1]);
            }
            data.add(shingle.stream().mapToDouble(d -> d).toArray());
            shingle.clear();
        }
        return data;
    }

    public DataFrame createAllDataForNycTaxi() throws ParseException {
        List<double[]> data = new ArrayList<>();
        DataFrame df = new DataFrame("/local/home/ylwu/code/os/ml/NAB/data/realKnownCause/nyc_taxi.csv");
        df.asType("timestamp", Date.class);
        df.asType("value", Double.class);
        List<Double> value = df.getColumnData("value");
//        System.out.println(Arrays.toString(value.toArray()));
        for (Double v : value) {
            data.add(new double[]{v});
        }
        return df;
    }


//    @Test
//    public void testNycTaxi() throws ParseException, IOException {
//        int shingleSize = 1;
//        Map<Integer, Float> shingleScoreThreshold = new HashMap<>();
//        shingleScoreThreshold.put(1, 2.0f);
//        shingleScoreThreshold.put(10, 1.7f);
//        shingleScoreThreshold.put(48, 1.2f);
//        int trainingDataSize = 512;
//        List<double[]> trainingData = createTrainingDataForNycTaxi(shingleSize, 512);
//        int expectedTrainingDataSize = trainingDataSize;
//        int anomalyTrainingDataSize = 0;
//        RandomCutForest forest = RandomCutForest.builder()
//                .numberOfTrees(200)
//                .sampleSize(trainingDataSize)
//                .outputAfter(trainingDataSize)
//                .dimensions(shingleSize) // still required! shingle can increase the dimension
//                .randomSeed(123)
//                .build();
//        double maxScore = Double.MIN_VALUE;
//        // train with first 1000 shingles
//        for (int i = 0; i < trainingDataSize ; i++) {
//            double[] point = trainingData.get(i);
//            double score = forest.getAnomalyScore(point);
//            maxScore = maxScore > score ? maxScore : score;
//            forest.update(point);
//        }
//        System.out.println(maxScore);
//
//        DateFormat simpleDateFormat = new SimpleDateFormat(YEAR_MONTH_DAY_HOUR_MIN_SECOND_FORMAT);
//        simpleDateFormat.setTimeZone(TimeZone.getTimeZone("America/New_York"));
//
//        DataFrame df = createAllDataForNycTaxi();
//        String index_name = "nyc_taxi_update_with_prediction_data_shingle"+shingleSize;
//        DataFrame bulkResult = new DataFrame();
//        bulkResult.addColumn("bulk_request");
//        List<Double> shingle = new ArrayList<>();
//        DataFrame resultDf = new DataFrame();
//        resultDf.addColumn("timestamp");
//        resultDf.addColumn("value");
//        resultDf.addColumn("score");
//        for (int m = trainingDataSize  + shingleSize; m < df.getSize()-shingleSize; m++) {
//            for (int j = m; j<m + shingleSize; j++) {
//                shingle.add((Double)df.getRowData(j)[1]);
//            }
//
//            double[] point = shingle.stream().mapToDouble(d -> d).toArray();
//            shingle.clear();
//            double score = forest.getAnomalyScore(point);
//            forest.update(point); // Update RCF with prediction data
//            bulkResult.addRow(ImmutableList.of(String.format("{ \"index\" : { \"_index\" : \"%s\", \"_id\" : \"%s\" } }",
//                    index_name, m)));
//            bulkResult.addRow(ImmutableList.of(String.format("{\"timestamp\":\"%s\",\"value\":%s,\"score\":%s,\"anomaly_type\":\"%s\"}",
//                    simpleDateFormat.format(df.getRowData(m)[0]), point[0], score, score > shingleScoreThreshold.get(shingleSize) ? "ANOMALOUS" : "EXPECTED")));
//            resultDf.addRow(ImmutableMap.of("timestamp", df.getRowData(m)[0], "value", point[0], "score", score));
//        }
//        resultDf.asType("timestamp", Date.class);
//        resultDf.asType("value", Double.class);
//        resultDf.asType("score", Double.class);
//        System.out.println("+++++++++++++++++++++++++++++ " + bulkResult.getSize());
//        //bulkResult.toCsv("/Users/ylwu/code/ylwu/MLPyGround/data/ad/nab/nyc_taxi_results_update_with_predict_data_shingle_size"+shingleSize);
//        System.out.println(resultDf.getSize());
//        System.out.println(resultDf.first(3));
//        resultDf = resultDf.sort("score", SortMethod.DOWN);
//        System.out.println(resultDf.first(300));
//    }

    @Test
    public void testNycTaxi() throws ParseException, IOException {
        int shingleSize = 1;
        int dimension = 1;
        Map<Integer, Float> shingleScoreThreshold = new HashMap<>();
        shingleScoreThreshold.put(1, 2.0f);
        shingleScoreThreshold.put(10, 1.7f);
        shingleScoreThreshold.put(48, 1.2f);
        int trainingDataSize = 512;
        DataFrame df = createAllDataForNycTaxi();
        int expectedTrainingDataSize = trainingDataSize;
        int anomalyTrainingDataSize = 0;
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder()
                .dimensions(shingleSize * dimension)
                .numberOfTrees(200)
                .sampleSize(trainingDataSize)
                .outputAfter(trainingDataSize)
                .parallelExecutionEnabled(false)
                .compact(true)
                .precision(Precision.FLOAT_32)
                .boundingBoxCacheFraction(1)
                .shingleSize(shingleSize)
                .anomalyRate(0.005)
                .internalShinglingEnabled(true)
                .build();
        double maxScore = Double.MIN_VALUE;

        DataFrame resultDf = new DataFrame();
        resultDf.addColumn("timestamp");
        resultDf.addColumn("value");
        resultDf.addColumn("score");
        resultDf.addColumn("anomaly_grade");

        // train with first 1000 shingles
        //List<Double> shingle = new ArrayList<>();
        for (int i = 0; i < df.getSize(); i++) {
//            for (int j=0;j<shingleSize;j++) {
//                double[] point = trainingData.get(i);
//            }
            Object[] rowData = df.getRowData(i);
            double[] point = new double[]{(Double)rowData[1]};
            Date time = (Date)rowData[0];
            long timestamp = time.getTime();

            AnomalyDescriptor process = forest.process(point, timestamp);
//            if (process.getRCFScore() > 0) {
//                System.out.println(i + ", " + process);
//            }
            resultDf.addRow(ImmutableMap.of("timestamp", rowData[0], "value", point[point.length - 1],
                    "score", process.getRCFScore(), "anomaly_grade", process.getAnomalyGrade(),
                    "anomaly_type", process.getAnomalyGrade() > 0? "ANOMALOUS":"EXPECTED"));
        }

        resultDf.asType("timestamp", Date.class);
        resultDf.asType("value", Double.class);
        resultDf.asType("score", Double.class);
        resultDf.asType("anomaly_grade", Double.class);
        resultDf.asType("anomaly_type", String.class);
//        System.out.println("+++++++++++++++++++++++++++++ " + bulkResult.getSize());
        //bulkResult.toCsv("/Users/ylwu/code/ylwu/MLPyGround/data/ad/nab/nyc_taxi_results_update_with_predict_data_shingle_size"+shingleSize);
        System.out.println(resultDf.getSize());
        System.out.println(resultDf.first(3));

        DataFrame bulkResult = new DataFrame();
        bulkResult.addColumn("bulk_request");
        String index_name = "nyc_taxi_trcf_results_shingle" + shingleSize;
        DateFormat simpleDateFormat = new SimpleDateFormat(YEAR_MONTH_DAY_HOUR_MIN_SECOND_FORMAT);
        simpleDateFormat.setTimeZone(TimeZone.getTimeZone("America/New_York"));
        for (int i = 0; i<resultDf.getSize(); i++) {
            Object[] rowData = resultDf.getRowData(i);
            bulkResult.addRow(ImmutableList.of(String.format("{ \"index\" : { \"_index\" : \"%s\", \"_id\" : \"%s\" } }",
                    index_name, i)));
            bulkResult.addRow(ImmutableList.of(String.format("{\"timestamp\":\"%s\",\"value\":%s,\"score\":%s,\"anomaly_grade\":%s,\"anomaly_type\":\"%s\"}",
                    simpleDateFormat.format(rowData[0]), rowData[1], rowData[2], rowData[3], rowData[4])));
//            resultDf.addRow(ImmutableMap.of("timestamp", df.getRowData(m)[0], "value", point[0], "score", score));
        }

        System.out.println("11111111111111111111111111111111");
        System.out.println(bulkResult.first(4));
        bulkResult.toCsv("/local/home/ylwu/code/os/ml/output/nyc_taxi_trcf_results_shingle_size"+shingleSize);


//        System.out.println("++++++++++++++++++++++++++++----------------------");
//        resultDf = resultDf.sort("score", SortMethod.DOWN);
//        System.out.println(resultDf.first(30));
//        resultDf.map(new MapFunction() {
//            @Override
//            public List calculate(DataFrame dataFrame, List<String> list) {
//                String col = list.get(0);
//
//                for (int i =  0;i<dataFrame.getSize(); i++) {
//                    dataFrame.getRowData(i)
//                }
//            }
//
//            @Override
//            public List<Class> getColumnDataType() {
//                return ImmutableList.of(String.class);
//            }
//
//            @Override
//            public List getColumnItemPos() {
//                return null;
//            }
//        }, ImmutableList.of("anomaly_grade"), ImmutableList.of("anomaly_type"));



//        System.out.println(maxScore);
//
//        DateFormat simpleDateFormat = new SimpleDateFormat(YEAR_MONTH_DAY_HOUR_MIN_SECOND_FORMAT);
//        simpleDateFormat.setTimeZone(TimeZone.getTimeZone("America/New_York"));
//
//
//        String index_name = "nyc_taxi_update_with_prediction_data_shingle" + shingleSize;
//        DataFrame bulkResult = new DataFrame();
//        bulkResult.addColumn("bulk_request");
//
//        DataFrame resultDf = new DataFrame();
//        resultDf.addColumn("timestamp");
//        resultDf.addColumn("value");
//        resultDf.addColumn("score");
//        for (int m = trainingDataSize + shingleSize; m < df.getSize() - shingleSize; m++) {
//            for (int j = m; j < m + shingleSize; j++) {
//                shingle.add((Double) df.getRowData(j)[1]);
//            }
//
//            double[] point = shingle.stream().mapToDouble(d -> d).toArray();
//            shingle.clear();
//            double score = forest.getAnomalyScore(point);
//            forest.update(point); // Update RCF with prediction data
//            bulkResult.addRow(ImmutableList.of(String.format("{ \"index\" : { \"_index\" : \"%s\", \"_id\" : \"%s\" } }",
//                    index_name, m)));
//            bulkResult.addRow(ImmutableList.of(String.format("{\"timestamp\":\"%s\",\"value\":%s,\"score\":%s,\"anomaly_type\":\"%s\"}",
//                    simpleDateFormat.format(df.getRowData(m)[0]), point[0], score, score > shingleScoreThreshold.get(shingleSize) ? "ANOMALOUS" : "EXPECTED")));
//            resultDf.addRow(ImmutableMap.of("timestamp", df.getRowData(m)[0], "value", point[0], "score", score));
//        }
//        resultDf.asType("timestamp", Date.class);
//        resultDf.asType("value", Double.class);
//        resultDf.asType("score", Double.class);
//        System.out.println("+++++++++++++++++++++++++++++ " + bulkResult.getSize());
//        //bulkResult.toCsv("/Users/ylwu/code/ylwu/MLPyGround/data/ad/nab/nyc_taxi_results_update_with_predict_data_shingle_size"+shingleSize);
//        System.out.println(resultDf.getSize());
//        System.out.println(resultDf.first(3));
//        resultDf = resultDf.sort("score", SortMethod.DOWN);
//        System.out.println(resultDf.first(300));
    }
}