package org.opensearch.ml.common.transport.tools;

import org.opensearch.action.ActionType;

public class MLGetToolsAction extends ActionType<MLToolsGetResponse> {
    public static final MLGetToolsAction INSTANCE = new MLGetToolsAction();
    public static final String NAME = "cluster:admin/opensearch/ml/tools/get";

    public MLGetToolsAction() {
        super(NAME, MLToolsGetResponse::new);
    }
}
