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

package org.opensearch.ml.common.transport.training;

import org.junit.Before;
import org.junit.Test;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataframe.DataFrameBuilder;
import org.opensearch.ml.common.input.dataset.DataFrameInputDataset;
import org.opensearch.ml.common.input.dataset.MLInputDataType;
import org.opensearch.ml.common.input.parameter.KMeansParams;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.input.MLInput;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Collections;
import java.util.HashMap;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;

public class MLTrainingTaskRequestTest {

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
    public void validate_Success() {
        MLTrainingTaskRequest request = MLTrainingTaskRequest.builder()
                .mlInput(mlInput)
                .build();
        assertNull(request.validate());
    }

    @Test
    public void writeTo() throws IOException {
        MLTrainingTaskRequest request = MLTrainingTaskRequest.builder()
            .mlInput(mlInput)
            .build();
        BytesStreamOutput bytesStreamOutput = new BytesStreamOutput();
        request.writeTo(bytesStreamOutput);
        request = new MLTrainingTaskRequest(bytesStreamOutput.bytes().streamInput());
        assertEquals(FunctionName.KMEANS, request.getInput().getAlgorithm());
        assertEquals(1, ((KMeansParams) request.getInput().getParameters()).getCentroids().intValue());
        assertEquals(MLInputDataType.DATA_FRAME, request.getInput().getInputDataset().getInputDataType());
    }

    @Test
    public void fromActionRequest_WithMLTrainingTaskRequest() {
        MLTrainingTaskRequest request = MLTrainingTaskRequest.builder()
            .mlInput(mlInput)
            .build();
        assertSame(request, MLTrainingTaskRequest.fromActionRequest(request));
    }

    @Test
    public void fromActionRequest_WithNonMLTrainingTaskRequest() {
        MLTrainingTaskRequest request = MLTrainingTaskRequest.builder()
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
        MLTrainingTaskRequest result = MLTrainingTaskRequest.fromActionRequest(actionRequest);
        assertNotSame(request, result);
        assertEquals(request.getInput().getAlgorithm(), result.getInput().getAlgorithm());
        assertEquals(request.getInput().getParameters(), result.getInput().getParameters());
        assertEquals(request.getInput().getInputDataset().getInputDataType(), result.getInput().getInputDataset().getInputDataType());
    }

    @Test(expected = UncheckedIOException.class)
    public void fromActionRequest_Exception() {
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
        MLTrainingTaskRequest.fromActionRequest(actionRequest);
    }
}