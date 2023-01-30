package org.opensearch.ml.engine.algorithms.time_series;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.timeseries.Forecast;
import ai.djl.timeseries.SampleForecast;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.timefeature.Lag;
import ai.djl.timeseries.timefeature.TimeFeature;
import ai.djl.timeseries.transform.InstanceSampler;
import ai.djl.timeseries.transform.PredictionSplitSampler;
import ai.djl.timeseries.transform.convert.Convert;
import ai.djl.timeseries.transform.feature.Feature;
import ai.djl.timeseries.transform.field.Field;
import ai.djl.timeseries.transform.split.Split;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.opensearch.ml.common.output.model.MLResultDataType;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensors;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;

import static org.opensearch.ml.engine.algorithms.time_series.TSForecastingModel.getTimeSeriesData;

/** The {@link Translator} for DeepAR time series forecasting tasks. */
public class DeepARTranslator implements Translator<Input, Output> {

    protected int predictionLength;
    protected int contextLength;

    protected String freq;

    private Batchifier batchifier;

    private boolean useFeatDynamicReal;
    private boolean useFeatStaticReal;
    private boolean useFeatStaticCat;
    private int historyLength;

    private static final String[] PRED_INPUT_FIELDS = {
            FieldName.FEAT_STATIC_CAT.name(),
            FieldName.FEAT_STATIC_REAL.name(),
            "PAST_" + FieldName.FEAT_TIME.name(),
            "PAST_" + FieldName.TARGET.name(),
            "PAST_" + FieldName.OBSERVED_VALUES.name(),
            "FUTURE_" + FieldName.FEAT_TIME.name()
    };

    private static final FieldName[] TIME_SERIES_FIELDS = {
            FieldName.FEAT_TIME, FieldName.OBSERVED_VALUES
    };

    private List<BiFunction<NDManager, List<LocalDateTime>, NDArray>> timeFeatures;
    private InstanceSampler instanceSampler;
    private String[] predictInputFields;

    /**
     * Constructs a new {@code DeepARTranslator} instance.
     *
     * @param builder the data to build with
     */
    public DeepARTranslator(Builder builder) {
        this.batchifier = builder.batchifier;
        this.freq = builder.freq;
        this.predictionLength = builder.predictionLength;
        // TODO: for inferring
        this.contextLength = builder.predictionLength;
        this.useFeatDynamicReal = builder.useFeatDynamicReal;
        this.useFeatStaticReal = builder.useFeatStaticReal;
        this.useFeatStaticCat = builder.useFeatStaticCat;

        List<Integer> lagsSeq = Lag.getLagsForFreq(freq);
        this.timeFeatures = TimeFeature.timeFeaturesFromFreqStr(freq);
        this.historyLength = contextLength + lagsSeq.get(lagsSeq.size() - 1);
        this.instanceSampler = PredictionSplitSampler.newTestSplitSampler();
        if (builder.useIsPad) {
            int len = PRED_INPUT_FIELDS.length;
            predictInputFields = new String[len + 1];
            System.arraycopy(PRED_INPUT_FIELDS, 0, predictInputFields, 0, len);
            predictInputFields[len] = "PAST_" + FieldName.IS_PAD.name();
        } else {
            predictInputFields = PRED_INPUT_FIELDS;
        }
    }

    /** {@inheritDoc} */
    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) {
        NDArray outputs = list.singletonOrThrow();
        TimeSeriesData data = (TimeSeriesData) ctx.getAttachment("input");
        outputs.attach((NDManager) ctx.getAttachment("manager"));
        Forecast forecast = new SampleForecast(outputs, data.getStartTime(), this.freq);

        List<ModelTensors> tensorOutoupt = new ArrayList<>();

        List<ModelTensor> tensors = new ArrayList<>();
        tensors.add(new ModelTensor("freq", forecast.freq()));

        Number[] forecastMedianData = forecast.median().toArray();
        tensors.add(new ModelTensor("forecast_median", forecastMedianData, new long[]{forecastMedianData.length}, MLResultDataType.FLOAT32, null));

        Number[] forecastMeanData = forecast.mean().toArray();
        tensors.add(new ModelTensor("forecast_mean", forecastMeanData, new long[]{forecastMeanData.length}, MLResultDataType.FLOAT32, null));

        Number[] forecastP99Data = forecast.quantile(0.99f).toArray();
        tensors.add(new ModelTensor("forecast_p99", forecastP99Data, new long[]{forecastP99Data.length}, MLResultDataType.FLOAT32, null));

        Number[] forecastP90Data = forecast.quantile(0.90f).toArray();
        tensors.add(new ModelTensor("forecast_p90", forecastP90Data, new long[]{forecastP90Data.length}, MLResultDataType.FLOAT32, null));

        Number[] forecastP50Data = forecast.quantile(0.50f).toArray();
        tensors.add(new ModelTensor("forecast_p50", forecastP50Data, new long[]{forecastP50Data.length}, MLResultDataType.FLOAT32, null));

        ModelTensors modelTensors = new ModelTensors(tensors);
        tensorOutoupt.add(modelTensors);

        Output output = new Output();
        ModelTensors modelTensorOutput = new ModelTensors(tensors);
        output.add(modelTensorOutput.toBytes());
        return output;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Input orgInput) {
        NDManager manager = ctx.getNDManager();
        TimeSeriesData input = null;
        try {
            input = getTimeSeriesData(manager, orgInput.getAsString(0));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        NDArray target = input.get(FieldName.TARGET);
        target.setName("target");

        ctx.setAttachment("input", input);
        ctx.setAttachment("manager", input.get(FieldName.TARGET).getManager());

        List<FieldName> removeFieldNames = new ArrayList<>();
        removeFieldNames.add(FieldName.FEAT_DYNAMIC_CAT);
        if (!useFeatStaticReal) {
            removeFieldNames.add(FieldName.FEAT_STATIC_REAL);
        }
        if (!useFeatDynamicReal) {
            removeFieldNames.add(FieldName.FEAT_DYNAMIC_REAL);
        }
        Field.removeFields(removeFieldNames, input);

        if (!useFeatStaticCat) {
            Field.setField(FieldName.FEAT_STATIC_CAT, manager.zeros(new Shape(1)), input);
        }

        if (!useFeatStaticReal) {
            Field.setField(FieldName.FEAT_STATIC_REAL, manager.zeros(new Shape(1)), input);
        }

        Convert.asArray(FieldName.FEAT_STATIC_CAT, 1, DataType.INT32, input);
        Convert.asArray(FieldName.FEAT_STATIC_REAL, 1, input);

        Feature.addObservedValuesIndicator(
                manager, FieldName.TARGET, FieldName.OBSERVED_VALUES, input);

        Feature.addTimeFeature(
                manager,
                FieldName.START,
                FieldName.TARGET,
                FieldName.FEAT_TIME,
                timeFeatures,
                predictionLength,
                freq,
                input);

        Feature.addAgeFeature(
                manager, FieldName.TARGET, FieldName.FEAT_AGE, predictionLength, input);

        FieldName[] inputFields;
        if (useFeatDynamicReal) {
            inputFields = new FieldName[3];
            inputFields[2] = FieldName.FEAT_DYNAMIC_REAL;
        } else {
            inputFields = new FieldName[2];
        }
        inputFields[0] = FieldName.FEAT_TIME;
        inputFields[1] = FieldName.FEAT_AGE;
        Convert.vstackFeatures(FieldName.FEAT_TIME, inputFields, input);

        Split.instanceSplit(
                manager,
                FieldName.TARGET,
                FieldName.IS_PAD,
                FieldName.START,
                FieldName.FORECAST_START,
                instanceSampler,
                historyLength,
                predictionLength,
                TIME_SERIES_FIELDS,
                0,
                input);

        input = Field.selectField(predictInputFields, input);

        return input.toNDList();
    }

    /**
     * Creates a builder to build a {@code DeepARTranslator}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a builder to build a {@code DeepARTranslator}.
     *
     * @param arguments the models' arguments
     * @return a new builder
     */
    public static Builder builder(Map<String, ?> arguments) {
        Builder builder = new Builder();

        builder.configPreProcess(arguments);
        builder.configPostProcess(arguments);

        return builder;
    }

    /** The builder for DeepAR translator. */
    public static class Builder {

        protected Batchifier batchifier = Batchifier.STACK;
        protected int predictionLength;

        protected String freq;

        // preProcess args
        boolean useFeatDynamicReal;
        boolean useFeatStaticReal;
        boolean useFeatStaticCat;
        boolean useIsPad;

        Builder() {}

        public Builder optBachifier(Batchifier batchifier) {
            this.batchifier = batchifier;
            return this;
        }

        /** {@inheritDoc} */
        protected void configPreProcess(Map<String, ?> arguments) {
            this.freq = ArgumentsUtil.stringValue(arguments, "freq", "D");
            this.predictionLength = ArgumentsUtil.intValue(arguments, "prediction_length");
            if (predictionLength <= 0) {
                throw new IllegalArgumentException(
                        "The value of `prediction_length` should be > 0");
            }
            if (arguments.containsKey("batchifier")) {
                batchifier = Batchifier.fromString((String) arguments.get("batchifier"));
            }
            useFeatDynamicReal =
                    ArgumentsUtil.booleanValue(
                            arguments,
                            "use_" + FieldName.FEAT_DYNAMIC_REAL.name().toLowerCase(),
                            false);
            useFeatStaticCat =
                    ArgumentsUtil.booleanValue(
                            arguments,
                            "use_" + FieldName.FEAT_STATIC_CAT.name().toLowerCase(),
                            false);
            useFeatStaticReal =
                    ArgumentsUtil.booleanValue(
                            arguments,
                            "use_" + FieldName.FEAT_STATIC_REAL.name().toLowerCase(),
                            false);
            useIsPad = ArgumentsUtil.booleanValue(arguments, "use_is_pad", true);
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public DeepARTranslator build() {
            return new DeepARTranslator(this);
        }

        public void configPostProcess(Map<String, ?> arguments) {

        }
    }
}
