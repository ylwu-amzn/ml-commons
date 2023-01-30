package org.opensearch.ml.engine.algorithms.time_series;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorFactory;
import com.google.gson.GsonBuilder;
import lombok.extern.log4j.Log4j2;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.dataset.MLInputDataset;
import org.opensearch.ml.common.dataset.TextDocsInputDataSet;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.model.MLModelConfig;
import org.opensearch.ml.common.output.model.ModelResultFilter;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.engine.algorithms.DLModel;
import org.opensearch.ml.engine.annotation.Function;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

@Log4j2
@Function(FunctionName.TIME_SERIES_FORECASTING)
public class TSForecastingModel extends DLModel {

    @Override
    public ModelTensorOutput predict(String modelId, MLInputDataset inputDataSet) throws TranslateException {
        try {
            TextDocsInputDataSet textDocsInput = (TextDocsInputDataSet) inputDataSet;
            ModelResultFilter resultFilter = textDocsInput.getResultFilter();
            List<String> docs = textDocsInput.getDocs();
            if (docs == null || docs.size() == 0) {
                throw new IllegalArgumentException("empty docs");
            }

            Input input = new Input();
            input.add(docs.get(0));
            Output output = getPredictor().predict(input);
            ModelTensors modelTensors = parseModelTensorOutput(output, resultFilter);
            List<ModelTensors> tensorOutputs = new ArrayList<>();
            tensorOutputs.add(modelTensors);
            return new ModelTensorOutput(tensorOutputs);
        } catch (Exception e) {
            throw new MLException("failed to predict ts forecasting");
        }
    }

    @Override
    public TranslatorFactory getTranslatorFactory(String engine, MLModelConfig modelConfig) {
        return new MyDeepARTranslatorFactory();
    }

    public static TimeSeriesData getTimeSeriesData(NDManager manager, String input) throws IOException {
        AirPassengers passengers =
                new GsonBuilder()
                        .setDateFormat("yyyy-MM")
                        .create()
                        .fromJson(input, AirPassengers.class);

        LocalDateTime start =
                passengers.start.toInstant().atZone(ZoneId.systemDefault()).toLocalDateTime();
        NDArray target = manager.create(passengers.target);
        TimeSeriesData data = new TimeSeriesData(10);
        data.setStartTime(start);
        data.setField(FieldName.TARGET, target);
        return data;
    }

    private static final class AirPassengers {
        Date start;
        float[] target;
    }
}
