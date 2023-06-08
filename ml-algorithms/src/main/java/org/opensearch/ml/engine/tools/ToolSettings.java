/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools;

import org.opensearch.common.settings.Setting;

public final class ToolSettings {

    private ToolSettings() {}

    public static final Setting<String> SUMMARY_MODEL_ID = Setting
        .simpleString("plugins.ml_commons.tools.summary_model", "Sh6OwYcBHgi99dxDOEtj", Setting.Property.NodeScope, Setting.Property.Dynamic);

}
