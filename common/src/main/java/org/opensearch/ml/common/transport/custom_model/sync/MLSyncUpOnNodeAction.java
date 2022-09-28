/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.custom_model.sync;

import org.opensearch.action.ActionType;

public class MLSyncUpOnNodeAction extends ActionType<MLSyncUpNodesResponse> {
    public static MLSyncUpOnNodeAction INSTANCE = new MLSyncUpOnNodeAction();
    public static final String NAME = "cluster:admin/opensearch/ml/sync_up_on_nodes";

    private MLSyncUpOnNodeAction() {
        super(NAME, MLSyncUpNodesResponse::new);
    }

}
