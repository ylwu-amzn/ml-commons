/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.model;

import org.opensearch.action.ActionType;

public class MLModelUpdateAction extends ActionType<MLModelUpdateResponse> {
    public static final MLModelUpdateAction INSTANCE = new MLModelUpdateAction();
    public static final String NAME = "cluster:admin/opensearch/ml/models/update";

    private MLModelUpdateAction() { super(NAME, MLModelUpdateResponse::new);}
}
