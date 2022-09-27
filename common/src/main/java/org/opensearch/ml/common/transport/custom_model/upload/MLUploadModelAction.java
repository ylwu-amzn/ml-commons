/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom_model.upload;

import org.opensearch.action.ActionType;
import org.opensearch.ml.common.transport.custom_model.load.LoadModelResponse;

public class MLUploadModelAction extends ActionType<UploadModelResponse> {
    public static MLUploadModelAction INSTANCE = new MLUploadModelAction();
    public static final String NAME = "cluster:admin/opensearch/ml/upload_custom_model";

    private MLUploadModelAction() {
        super(NAME, UploadModelResponse::new);
    }

}
