package org.opensearch.ml.settings;

import org.opensearch.common.settings.Setting;

public final class MLCommonsSettings {

    private MLCommonsSettings() {}

    public static final Setting<String> ML_COMMONS_TASK_DISPATCH_POLICY = Setting
            .simpleString(
                    "plugins.ml_commons.task_dispatch_policy",
                    "round_robin",
                    Setting.Property.NodeScope,
                    Setting.Property.Dynamic
            );
}
