/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.memory;

import org.opensearch.action.ActionType;

/**
 * Action to list entities in memory container graph data
 */
public class MLListGraphEntitiesAction extends ActionType<MLGraphSearchResponse> {

    public static final MLListGraphEntitiesAction INSTANCE = new MLListGraphEntitiesAction();
    public static final String NAME = "cluster:admin/opensearch/ml/memory_containers/graph/entities/list";

    private MLListGraphEntitiesAction() {
        super(NAME, MLGraphSearchResponse::new);
    }
}