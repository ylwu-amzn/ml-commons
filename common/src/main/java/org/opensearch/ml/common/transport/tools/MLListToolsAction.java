package org.opensearch.ml.common.transport.tools;

import org.opensearch.action.ActionType;

public class MLListToolsAction extends ActionType<MLToolsListResponse> {
    public static final MLListToolsAction INSTANCE = new MLListToolsAction();
    public static final String NAME = "cluster:admin/opensearch/ml/tools/get";

    public MLListToolsAction() {
        super(NAME, MLToolsListResponse::new);
    }
}
