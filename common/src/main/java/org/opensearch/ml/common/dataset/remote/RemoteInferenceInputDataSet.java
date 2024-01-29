/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.dataset.remote;

import java.io.IOException;
import java.util.Map;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.ml.common.annotation.InputDataSet;
import org.opensearch.ml.common.dataset.MLInputDataType;
import org.opensearch.ml.common.dataset.MLInputDataset;
import org.opensearch.ml.common.utils.StringUtils;

@Getter
@InputDataSet(MLInputDataType.REMOTE)
public class RemoteInferenceInputDataSet extends MLInputDataset {

    @Setter
    private Map<String, String> parameters;

    @Builder(toBuilder = true)
    public RemoteInferenceInputDataSet(Map<String, String> parameters) {
        super(MLInputDataType.REMOTE);
        this.parameters = parameters;
    }

    public RemoteInferenceInputDataSet(StreamInput streamInput) throws IOException {
        super(MLInputDataType.REMOTE);
        if (streamInput.readBoolean()) {
            parameters = streamInput.readMap(s -> s.readString(), s-> s.readString());
        }
    }

    @Override
    public void writeTo(StreamOutput streamOutput) throws IOException {
        super.writeTo(streamOutput);
        if (parameters !=  null) {
            streamOutput.writeBoolean(true);
            streamOutput.writeMap(parameters, StreamOutput::writeString, StreamOutput::writeString);
        } else {
            streamOutput.writeBoolean(false);
        }
    }

//    @Setter
//    private Map<String, Object> rawParameters;
//
//    @Builder
//    public RemoteInferenceInputDataSet(Map<String, String> parameters) {
//        super(MLInputDataType.REMOTE);
//        this.parameters = parameters;
//    }
//
//    public RemoteInferenceInputDataSet(StreamInput streamInput) throws IOException {
//        super(MLInputDataType.REMOTE);
//        if (streamInput.readBoolean()) {
//            System.out.println("ylwudebug111 --- read from input stream " );
//            rawParameters = streamInput.readMap();
//            parameters = StringUtils.getParameterMap(rawParameters);
//        }
//    }
//
//    @Override
//    public void writeTo(StreamOutput streamOutput) throws IOException {
//        super.writeTo(streamOutput);
//        if (rawParameters !=  null) {
//            System.out.println("ylwudebug111 --- write to stream " );
//            streamOutput.writeBoolean(true);
//            streamOutput.writeMap(rawParameters);
//        } else {
//            streamOutput.writeBoolean(false);
//        }
//    }

}
