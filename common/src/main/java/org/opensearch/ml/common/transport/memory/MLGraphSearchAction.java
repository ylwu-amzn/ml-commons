/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.memory;

import org.opensearch.action.ActionType;

/**
 * Action to search memory container graph data
 */
public class MLGraphSearchAction extends ActionType<MLGraphSearchResponse> {

    public static final MLGraphSearchAction INSTANCE = new MLGraphSearchAction();
    public static final String NAME = "cluster:admin/opensearch/ml/memory_containers/graph/search";

    private MLGraphSearchAction() {
        super(NAME, MLGraphSearchResponse::new);
    }
}