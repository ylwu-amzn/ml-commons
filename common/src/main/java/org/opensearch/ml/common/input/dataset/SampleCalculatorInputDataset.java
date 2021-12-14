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

package org.opensearch.ml.common.input.dataset;

import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.FieldDefaults;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.ml.common.annotation.InputDataSet;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataframe.DataFrameType;
import org.opensearch.ml.common.dataframe.DefaultDataFrame;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * DataFrame based input data. Client directly passes the data frame to ML plugin with this.
 */
@Getter
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
@InputDataSet(MLInputDataType.SAMPLE_DATA)
public class SampleCalculatorInputDataset extends MLInputDataset {
    String operation;
    List<Double> inputData;

    @Builder
    public SampleCalculatorInputDataset(@NonNull String operation, @NonNull List<Double> inputData) {
        super(MLInputDataType.SAMPLE_DATA);
        this.operation = operation;
        this.inputData = inputData;
    }

    public SampleCalculatorInputDataset(StreamInput in) throws IOException {
        super(MLInputDataType.SAMPLE_DATA);
        operation = in.readString();
        inputData = new ArrayList<>();
        if (in.readBoolean()) {
            int size = in.readInt();
            for (int i = 0; i < size; i++) {
                inputData.add(in.readDouble());
            }
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(operation);
        if (inputData != null && inputData.size() > 0) {
            out.writeBoolean(true);
            out.writeInt(inputData.size());
            for (Double data : inputData) {
                out.writeDouble(data);
            }
        }
    }
}
