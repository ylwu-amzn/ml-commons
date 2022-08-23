/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom.uploadchunk;

import org.opensearch.action.ActionType;
import org.opensearch.ml.common.transport.custom.load.LoadModelResponse;

public class MLUploadModelChunkAction extends ActionType<LoadModelResponse> {
    public static MLUploadModelChunkAction INSTANCE = new MLUploadModelChunkAction();
    public static final String NAME = "cluster:admin/opensearch/ml/upload_custom_model_chunk";

    private MLUploadModelChunkAction() {
        super(NAME, LoadModelResponse::new);
    }

}
