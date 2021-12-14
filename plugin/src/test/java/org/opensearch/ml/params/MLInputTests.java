/*
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  The OpenSearch Contributors require contributions made to
 *  this file be licensed under the Apache-2.0 license or a
 *  compatible open source license.
 *
 *  Modifications Copyright OpenSearch Contributors. See
 *  GitHub history for details.
 */

package org.opensearch.ml.params;

import static org.opensearch.ml.utils.TestHelper.parser;

import java.io.IOException;

import org.opensearch.common.xcontent.XContentParser;
import org.opensearch.ml.common.dataframe.ColumnType;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.dataset.DataFrameInputDataset;
import org.opensearch.ml.common.input.dataset.SearchQueryInputDataset;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.test.OpenSearchTestCase;

public class MLInputTests extends OpenSearchTestCase {

    public void testParseKmeansInputQuery() throws IOException {
        String query =
            "{\"input_query\":{\"query\":{\"bool\":{\"filter\":[{\"term\":{\"k1\":1}}]}},\"size\":10},\"input_index\":[\"test_data\"]}";
        XContentParser parser = parser(query);
        MLInput mlInput = MLInput.parse(parser, FunctionName.KMEANS.name());
        String expectedQuery =
            "{\"size\":10,\"query\":{\"bool\":{\"filter\":[{\"term\":{\"k1\":{\"value\":1,\"boost\":1.0}}}],\"adjust_pure_negative\":true,\"boost\":1.0}}}";
        SearchQueryInputDataset inputDataset = (SearchQueryInputDataset) mlInput.getInputDataset();
        assertEquals(expectedQuery, inputDataset.getSearchSourceBuilder().toString());
    }

    public void testParseKmeansInputDataFrame() throws IOException {
        String query = "{\"input_data\":{\"column_metas\":[{\"name\":\"total_sum\",\"column_type\":\"DOUBLE\"},{\"name\":\"is_error\","
            + "\"column_type\":\"BOOLEAN\"}],\"rows\":[{\"values\":[{\"column_type\":\"DOUBLE\",\"value\":15},"
            + "{\"column_type\":\"BOOLEAN\",\"value\":false}]},{\"values\":[{\"column_type\":\"DOUBLE\",\"value\":100},"
            + "{\"column_type\":\"BOOLEAN\",\"value\":true}]}]}}";
        XContentParser parser = parser(query);
        MLInput mlInput = MLInput.parse(parser, FunctionName.KMEANS.name());
        DataFrameInputDataset inputDataset = (DataFrameInputDataset) mlInput.getInputDataset();
        DataFrame dataFrame = inputDataset.getDataFrame();

        assertEquals(2, dataFrame.columnMetas().length);
        assertEquals(ColumnType.DOUBLE, dataFrame.columnMetas()[0].getColumnType());
        assertEquals(ColumnType.BOOLEAN, dataFrame.columnMetas()[1].getColumnType());
        assertEquals("total_sum", dataFrame.columnMetas()[0].getName());
        assertEquals("is_error", dataFrame.columnMetas()[1].getName());

        assertEquals(ColumnType.DOUBLE, dataFrame.getRow(0).getValue(0).columnType());
        assertEquals(ColumnType.BOOLEAN, dataFrame.getRow(0).getValue(1).columnType());
        assertEquals(15.0, dataFrame.getRow(0).getValue(0).getValue());
        assertEquals(false, dataFrame.getRow(0).getValue(1).getValue());
    }

}
