/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.memory;

import org.opensearch.action.ActionType;
import org.opensearch.action.delete.DeleteResponse;

/**
 * Action to delete graph data from a memory container
 */
public class MLDeleteGraphDataAction extends ActionType<DeleteResponse> {

    public static final MLDeleteGraphDataAction INSTANCE = new MLDeleteGraphDataAction();
    public static final String NAME = "cluster:admin/opensearch/ml/memory_containers/graph/delete";

    private MLDeleteGraphDataAction() {
        super(NAME, DeleteResponse::new);
    }
}