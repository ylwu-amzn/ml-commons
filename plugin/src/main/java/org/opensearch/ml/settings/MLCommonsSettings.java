/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.settings;

import java.util.List;
import java.util.function.Function;

import org.opensearch.common.settings.Setting;

import com.google.common.collect.ImmutableList;

public final class MLCommonsSettings {

    private MLCommonsSettings() {}

    public static final Setting<String> ML_COMMONS_TASK_DISPATCH_POLICY = Setting
        .simpleString("plugins.ml_commons.task_dispatch_policy", "round_robin", Setting.Property.NodeScope, Setting.Property.Dynamic);

    public static final Setting<Integer> ML_COMMONS_MAX_MODELS_PER_NODE = Setting
        .intSetting("plugins.ml_commons.max_model_on_node", 10, 0, 10000, Setting.Property.NodeScope, Setting.Property.Dynamic);
    public static final Setting<Integer> ML_COMMONS_MAX_REGISTER_MODEL_TASKS_PER_NODE = Setting
        .intSetting(
            "plugins.ml_commons.max_register_model_tasks_per_node",
            10,
            0,
            10,
            Setting.Property.NodeScope,
            Setting.Property.Dynamic
        );
    public static final Setting<Integer> ML_COMMONS_MAX_DEPLOY_MODEL_TASKS_PER_NODE = Setting
        .intSetting("plugins.ml_commons.max_deploy_model_tasks_per_node", 10, 0, 10, Setting.Property.NodeScope, Setting.Property.Dynamic);
    public static final Setting<Integer> ML_COMMONS_MAX_ML_TASK_PER_NODE = Setting
        .intSetting("plugins.ml_commons.max_ml_task_per_node", 10, 0, 10000, Setting.Property.NodeScope, Setting.Property.Dynamic);
    public static final Setting<Boolean> ML_COMMONS_ONLY_RUN_ON_ML_NODE = Setting
        .boolSetting("plugins.ml_commons.only_run_on_ml_node", true, Setting.Property.NodeScope, Setting.Property.Dynamic);

    public static final Setting<Boolean> ML_COMMONS_ENABLE_INHOUSE_PYTHON_MODEL = Setting
        .boolSetting("plugins.ml_commons.enable_inhouse_python_model", false, Setting.Property.NodeScope, Setting.Property.Dynamic);
    public static final Setting<Integer> ML_COMMONS_SYNC_UP_JOB_INTERVAL_IN_SECONDS = Setting
        .intSetting(
            "plugins.ml_commons.sync_up_job_interval_in_seconds",
            10,
            0,
            86400,
            Setting.Property.NodeScope,
            Setting.Property.Dynamic
        );

    public static final Setting<Integer> ML_COMMONS_ML_TASK_TIMEOUT_IN_SECONDS = Setting
        .intSetting("plugins.ml_commons.ml_task_timeout_in_seconds", 600, 1, 86400, Setting.Property.NodeScope, Setting.Property.Dynamic);
    public static final Setting<Long> ML_COMMONS_MONITORING_REQUEST_COUNT = Setting
        .longSetting(
            "plugins.ml_commons.monitoring_request_count",
            100,
            0,
            10_000_000,
            Setting.Property.NodeScope,
            Setting.Property.Dynamic
        );

    public static final Setting<String> ML_COMMONS_TRUSTED_URL_REGEX = Setting
        .simpleString(
            "plugins.ml_commons.trusted_url_regex",
            "^(https?|ftp|file)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]",
            Setting.Property.NodeScope,
            Setting.Property.Dynamic
        );

    public static final Setting<Integer> ML_COMMONS_NATIVE_MEM_THRESHOLD = Setting
        .intSetting("plugins.ml_commons.native_memory_threshold", 90, 0, 100, Setting.Property.NodeScope, Setting.Property.Dynamic);

    public static final Setting<Integer> ML_COMMONS_JVM_HEAP_MEM_THRESHOLD = Setting
        .intSetting("plugins.ml_commons.jvm_heap_memory_threshold", 85, 0, 100, Setting.Property.NodeScope, Setting.Property.Dynamic);

    public static final Setting<String> ML_COMMONS_EXCLUDE_NODE_NAMES = Setting
        .simpleString("plugins.ml_commons.exclude_nodes._name", Setting.Property.NodeScope, Setting.Property.Dynamic);
    public static final Setting<Boolean> ML_COMMONS_ALLOW_CUSTOM_DEPLOYMENT_PLAN = Setting
        .boolSetting("plugins.ml_commons.allow_custom_deployment_plan", false, Setting.Property.NodeScope, Setting.Property.Dynamic);

    public static final Setting<Boolean> ML_COMMONS_MODEL_AUTO_REDEPLOY_ENABLE = Setting
        .boolSetting("plugins.ml_commons.model_auto_redeploy.enable", false, Setting.Property.NodeScope, Setting.Property.Dynamic);

    public static final Setting<Integer> ML_COMMONS_MODEL_AUTO_REDEPLOY_LIFETIME_RETRY_TIMES = Setting
        .intSetting("plugins.ml_commons.model_auto_redeploy.lifetime_retry_times", 3, Setting.Property.NodeScope, Setting.Property.Dynamic);

    public static final Setting<Float> ML_COMMONS_MODEL_AUTO_REDEPLOY_SUCCESS_RATIO = Setting
        .floatSetting(
            "plugins.ml_commons.model_auto_redeploy_success_ratio",
            0.8f,
            0f,
            1f,
            Setting.Property.NodeScope,
            Setting.Property.Dynamic
        );

    // This setting is to enable/disable model url in model register API.
    public static final Setting<Boolean> ML_COMMONS_ALLOW_MODEL_URL = Setting
        .boolSetting("plugins.ml_commons.allow_registering_model_via_url", false, Setting.Property.NodeScope, Setting.Property.Dynamic);

    public static final Setting<Boolean> ML_COMMONS_ALLOW_LOCAL_FILE_UPLOAD = Setting
        .boolSetting(
            "plugins.ml_commons.allow_registering_model_via_local_file",
            false,
            Setting.Property.NodeScope,
            Setting.Property.Dynamic
        );

    public static final Setting<Boolean> ML_COMMONS_MODEL_ACCESS_CONTROL_ENABLED = Setting
        .boolSetting("plugins.ml_commons.model_access_control_enabled", false, Setting.Property.NodeScope, Setting.Property.Dynamic);

    public static final Setting<Boolean> ML_COMMONS_CONNECTOR_ACCESS_CONTROL_ENABLED = Setting
        .boolSetting("plugins.ml_commons.connector_access_control_enabled", false, Setting.Property.NodeScope, Setting.Property.Dynamic);

    public static final Setting<List<String>> ML_COMMONS_TRUSTED_CONNECTOR_ENDPOINTS_REGEX = Setting
        .listSetting(
            "plugins.ml_commons.trusted_connector_endpoints_regex",
            ImmutableList
                .of(
                    "^https://runtime\\.sagemaker\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
                    "^https://api\\.openai\\.com/.*$",
                    "^https://api\\.cohere\\.ai/.*$"
                ),
            Function.identity(),
            Setting.Property.NodeScope,
            Setting.Property.Dynamic
        );

    public static final Setting<List<String>> ML_COMMONS_DISABLED_FEATURE = Setting
        .listSetting(
            "plugins.ml_commons.disabled_feature",
            ImmutableList.of(),
            Function.identity(),
            Setting.Property.NodeScope,
            Setting.Property.Dynamic
        );

    public static final Setting<List<String>> ML_COMMONS_TASK_DISPATCHER_REMOTE_MODEL = Setting
        .listSetting(
            "plugins.ml_commons.task_dispatcher.eligible_node_role.remote_model",
            ImmutableList.of("data", "ml"),
            Function.identity(),
            Setting.Property.NodeScope,
            Setting.Property.Dynamic
        );

    public static final Setting<List<String>> ML_COMMONS_TASK_DISPATCHER_LOCAL_MODEL = Setting
        .listSetting(
            "plugins.ml_commons.task_dispatcher.eligible_node_role.local_model",
            ImmutableList.of("ml"),
            Function.identity(),
            Setting.Property.NodeScope,
            Setting.Property.Dynamic
        );
}
