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

package org.opensearch.ml.common.input;

import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.common.xcontent.ToXContentObject;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.parameter.Parameters;
import org.opensearch.ml.common.input.dataset.MLInputDataset;

import java.io.IOException;

public interface Input extends ToXContentObject, Writeable {

    FunctionName getFunctionName();
    void setFunctionName(FunctionName functionName);

    @Override
    default void writeTo(StreamOutput out) throws IOException {
        out.writeEnum(getFunctionName());
    }

    default MLInputDataSet getInputDataset() {
        throw new IllegalArgumentException("Don't support get input data set");
    }

    default Parameters getParameters() {
        throw new IllegalArgumentException("Don't support get input data set");
    };
}
