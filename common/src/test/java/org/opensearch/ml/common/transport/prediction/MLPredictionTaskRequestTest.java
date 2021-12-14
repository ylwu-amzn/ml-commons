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

package org.opensearch.ml.common.transport.prediction;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Collections;
import java.util.HashMap;

import lombok.NonNull;
import org.junit.Before;
import org.junit.Test;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.ml.common.dataframe.ColumnType;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataframe.DataFrameBuilder;
import org.opensearch.ml.common.input.dataset.DataFrameInputDataset;
import org.opensearch.ml.common.input.dataset.MLInputDataType;
import org.opensearch.ml.common.input.dataset.MLInputDataset;
import org.opensearch.ml.common.input.dataset.SearchQueryInputDataset;
import org.opensearch.ml.common.input.parameter.KMeansParams;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.search.builder.SearchSourceBuilder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;

public class MLPredictionTaskRequestTest {

    private MLInput mlInput;

    @Before
    public void setUp() {
        DataFrame dataFrame = DataFrameBuilder.load(Collections.singletonList(new HashMap<String, Object>() {{
            put("key1", 2.0D);
        }}));
        mlInput = MLInput.builder()
                .algorithm(FunctionName.KMEANS)
                .parameters(KMeansParams.builder().centroids(1).build())
                .inputDataset(DataFrameInputDataset.builder().dataFrame(dataFrame).build())
                .build();
    }

    @Test
    public void writeTo_Success() throws IOException {

        MLPredictionTaskRequest request = MLPredictionTaskRequest.builder()
            .mlInput(mlInput)
            .build();
        BytesStreamOutput bytesStreamOutput = new BytesStreamOutput();
        request.writeTo(bytesStreamOutput);
        request = new MLPredictionTaskRequest(bytesStreamOutput.bytes().streamInput());
        assertEquals(FunctionName.KMEANS, request.getInput().getAlgorithm());
        KMeansParams params = (KMeansParams)request.getInput().getParameters();
        assertEquals(1, params.getCentroids().intValue());
        MLInputDataset inputDataset = request.getInput().getInputDataset();
        assertEquals(MLInputDataType.DATA_FRAME, inputDataset.getInputDataType());
        DataFrame dataFrame = ((DataFrameInputDataset) inputDataset).getDataFrame();
        assertEquals(1, dataFrame.size());
        assertEquals(1, dataFrame.columnMetas().length);
        assertEquals("key1", dataFrame.columnMetas()[0].getName());
        assertEquals(ColumnType.DOUBLE, dataFrame.columnMetas()[0].getColumnType());
        assertEquals(1, dataFrame.getRow(0).size());
        assertEquals(2.00, dataFrame.getRow(0).getValue(0).getValue());

        assertNull(request.getModelId());
    }

    @Test
    public void validate_Success() {
        MLPredictionTaskRequest request = MLPredictionTaskRequest.builder()
            .mlInput(mlInput)
            .build();

        assertNull(request.validate());
    }

    @Test
    public void validate_Exception_NullMLInput() {
        mlInput.setAlgorithm(null);
        MLPredictionTaskRequest request = MLPredictionTaskRequest.builder()
                .build();

        ActionRequestValidationException exception = request.validate();
        assertEquals("Validation Failed: 1: ML input can't be null;", exception.getMessage());
    }

    @Test
    public void validate_Exception_NullInputDataset() {
        mlInput.setInputDataset(null);
        MLPredictionTaskRequest request = MLPredictionTaskRequest.builder()
                .mlInput(mlInput)
                .build();

        ActionRequestValidationException exception = request.validate();

        assertEquals("Validation Failed: 1: input data can't be null;", exception.getMessage());
    }


    @Test
    public void fromActionRequest_Success_WithMLPredictionTaskRequest() {
        MLPredictionTaskRequest request = MLPredictionTaskRequest.builder()
                .mlInput(mlInput)
                .build();
        assertSame(MLPredictionTaskRequest.fromActionRequest(request), request);
    }

    @Test
    public void fromActionRequest_Success_WithNonMLPredictionTaskRequest_DataFrameInput() {
        fromActionRequest_Success_WithNonMLPredictionTaskRequest(mlInput);
    }

    @Test
    public void fromActionRequest_Success_WithNonMLPredictionTaskRequest_SearchQueryInput() {
        @NonNull SearchSourceBuilder searchSourceBuilder = new SearchSourceBuilder();
        searchSourceBuilder.query(new MatchAllQueryBuilder());
        mlInput.setInputDataset(SearchQueryInputDataset.builder()
                .indices(Collections.singletonList("test_index"))
                .searchSourceBuilder(searchSourceBuilder)
                .build());
        fromActionRequest_Success_WithNonMLPredictionTaskRequest(mlInput);
    }

    private void fromActionRequest_Success_WithNonMLPredictionTaskRequest(MLInput mlInput) {
        MLPredictionTaskRequest request = MLPredictionTaskRequest.builder()
                .mlInput(mlInput)
                .build();
        ActionRequest actionRequest = new ActionRequest() {
            @Override
            public ActionRequestValidationException validate() {
                return null;
            }

            @Override
            public void writeTo(StreamOutput out) throws IOException {
                request.writeTo(out);
            }
        };
        MLPredictionTaskRequest result = MLPredictionTaskRequest.fromActionRequest(actionRequest);
        assertNotSame(result, request);
        assertEquals(request.getInput().getAlgorithm(), result.getInput().getAlgorithm());
        assertEquals(request.getInput().getInputDataset().getInputDataType(), result.getInput().getInputDataset().getInputDataType());
    }

    @Test(expected = UncheckedIOException.class)
    public void fromActionRequest_IOException() {
        ActionRequest actionRequest = new ActionRequest() {
            @Override
            public ActionRequestValidationException validate() {
                return null;
            }

            @Override
            public void writeTo(StreamOutput out) throws IOException {
                throw new IOException("test");
            }
        };
        MLPredictionTaskRequest.fromActionRequest(actionRequest);
    }
}