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

package org.opensearch.ml.common.output;

import org.junit.Before;
import org.junit.Test;
import org.opensearch.common.Strings;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.ml.common.dataframe.ColumnMeta;
import org.opensearch.ml.common.dataframe.ColumnType;
import org.opensearch.ml.common.dataframe.ColumnValue;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataframe.DefaultDataFrame;
import org.opensearch.ml.common.dataframe.IntValue;
import org.opensearch.ml.common.dataframe.Row;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class MLPredictionOutputTest {

    MLPredictionOutput output;
    @Before
    public void setUp() {
        ColumnMeta[] columnMetas = new ColumnMeta[]{new ColumnMeta("test", ColumnType.INTEGER)};
        List<Row> rows = new ArrayList<>();
        rows.add(new Row(new ColumnValue[]{new IntValue(1)}));
        rows.add(new Row(new ColumnValue[]{new IntValue(2)}));
        DataFrame dataFrame = new DefaultDataFrame(columnMetas, rows);
        output = MLPredictionOutput.builder()
                .taskId("test_task_id")
                .status("test_status")
                .predictionResult(dataFrame)
                .build();
    }
    @Test
    public void toXContent() throws IOException {
        XContentBuilder builder = XContentFactory.contentBuilder(XContentType.JSON);
        output.toXContent(builder, ToXContent.EMPTY_PARAMS);
        String jsonStr = Strings.toString(builder);
        assertEquals("{\"task_id\":\"test_task_id\",\"status\":\"test_status\",\"prediction_result\":" +
                "{\"column_metas\":[{\"name\":\"test\",\"column_type\":\"INTEGER\"}],\"rows\":[{\"values\":" +
                "[{\"column_type\":\"INTEGER\",\"value\":1}]},{\"values\":[{\"column_type\":\"INTEGER\"," +
                "\"value\":2}]}]}}", jsonStr);
    }

    @Test
    public void toXContent_EmptyOutput() throws IOException {
        MLPredictionOutput output = MLPredictionOutput.builder().build();
        XContentBuilder builder = XContentFactory.contentBuilder(XContentType.JSON);
        output.toXContent(builder, ToXContent.EMPTY_PARAMS);
        String jsonStr = Strings.toString(builder);
        assertEquals("{}", jsonStr);
    }

    @Test
    public void readInputStream_Success() throws IOException {
        readInputStream(output);
    }

    @Test
    public void readInputStream_Success_EmptyPredictResult() throws IOException {
        output.setPredictionResult(null);
        readInputStream(output);
    }

    private void readInputStream(MLPredictionOutput output) throws IOException {
        BytesStreamOutput bytesStreamOutput = new BytesStreamOutput();
        output.writeTo(bytesStreamOutput);

        StreamInput streamInput = bytesStreamOutput.bytes().streamInput();
        OutputType outputType = streamInput.readEnum(OutputType.class);
        assertEquals(OutputType.PREDICTION, outputType);
        MLPredictionOutput parsedOutput = new MLPredictionOutput(streamInput);
        assertEquals(output.getType(), parsedOutput.getType());
        assertEquals(output.getTaskId(), parsedOutput.getTaskId());
        assertEquals(output.getStatus(), parsedOutput.getStatus());
        if (output.predictionResult == null) {
            assertEquals(output.predictionResult, parsedOutput.getPredictionResult());
        } else {
            assertEquals(output.predictionResult.size(), parsedOutput.getPredictionResult().size());
            for (int i = 0 ;i<output.predictionResult.size(); i++) {
                Row row = output.predictionResult.getRow(i);
                Row parsedRow = parsedOutput.predictionResult.getRow(i);
                assertEquals(row, parsedRow);
            }
        }
    }
}
