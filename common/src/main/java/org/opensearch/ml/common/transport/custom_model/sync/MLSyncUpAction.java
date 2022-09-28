/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom_model.sync;

import org.opensearch.action.ActionType;

public class MLSyncUpAction extends ActionType<MLSyncUpResponse> {
    public static MLSyncUpAction INSTANCE = new MLSyncUpAction();
    public static final String NAME = "cluster:admin/opensearch/ml/sync_up";

    private MLSyncUpAction() {
        super(NAME, MLSyncUpResponse::new);
    }

}
