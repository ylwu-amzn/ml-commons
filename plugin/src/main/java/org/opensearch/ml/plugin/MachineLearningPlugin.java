/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.plugin;

import static org.opensearch.ml.common.CommonValue.ML_MODEL_INDEX;
import static org.opensearch.ml.common.CommonValue.ML_TASK_INDEX;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionResponse;
import org.opensearch.client.Client;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.node.DiscoveryNodes;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.io.stream.NamedWriteableRegistry;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.IndexScopedSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.settings.SettingsFilter;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.env.Environment;
import org.opensearch.env.NodeEnvironment;
import org.opensearch.ml.action.execute.TransportExecuteTaskAction;
import org.opensearch.ml.action.handler.MLSearchHandler;
import org.opensearch.ml.action.models.DeleteModelTransportAction;
import org.opensearch.ml.action.models.GetModelTransportAction;
import org.opensearch.ml.action.models.SearchModelTransportAction;
import org.opensearch.ml.action.models.upload_chunk.MLModelChunkUploader;
import org.opensearch.ml.action.models.upload_chunk.MLModelMetaUploader;
import org.opensearch.ml.action.models.upload_chunk.TransportUploadModelChunkAction;
import org.opensearch.ml.action.models.upload_chunk.TransportUploadModelMetaAction;
import org.opensearch.ml.action.prediction.TransportPredictionTaskAction;
import org.opensearch.ml.action.profile.MLProfileAction;
import org.opensearch.ml.action.profile.MLProfileTransportAction;
import org.opensearch.ml.action.stats.MLStatsNodesAction;
import org.opensearch.ml.action.stats.MLStatsNodesTransportAction;
import org.opensearch.ml.action.tasks.DeleteTaskTransportAction;
import org.opensearch.ml.action.tasks.GetTaskTransportAction;
import org.opensearch.ml.action.tasks.SearchTaskTransportAction;
import org.opensearch.ml.action.trained_model.forward.TransportForwardAction;
import org.opensearch.ml.action.trained_model.load.TransportLoadModelAction;
import org.opensearch.ml.action.trained_model.load.TransportLoadModelOnNodeAction;
import org.opensearch.ml.action.trained_model.syncup.TransportSyncUpOnNodeAction;
import org.opensearch.ml.action.trained_model.unload.TransportUnloadModelAction;
import org.opensearch.ml.action.trained_model.upload.MLModelUploader;
import org.opensearch.ml.action.trained_model.upload.TransportUploadModelAction;
import org.opensearch.ml.action.training.TransportTrainingTaskAction;
import org.opensearch.ml.action.trainpredict.TransportTrainAndPredictionTaskAction;
import org.opensearch.ml.cluster.DiscoveryNodeHelper;
import org.opensearch.ml.cluster.MLCommonsClusterEventListener;
import org.opensearch.ml.cluster.MLCommonsClusterManagerEventListener;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.breaker.MLCircuitBreakerService;
import org.opensearch.ml.common.input.execute.anomalylocalization.AnomalyLocalizationInput;
import org.opensearch.ml.common.input.execute.samplecalculator.LocalSampleCalculatorInput;
import org.opensearch.ml.common.input.parameter.ad.AnomalyDetectionLibSVMParams;
import org.opensearch.ml.common.input.parameter.clustering.KMeansParams;
import org.opensearch.ml.common.input.parameter.clustering.RCFSummarizeParams;
import org.opensearch.ml.common.input.parameter.rcf.BatchRCFParams;
import org.opensearch.ml.common.input.parameter.rcf.FitRCFParams;
import org.opensearch.ml.common.input.parameter.regression.LinearRegressionParams;
import org.opensearch.ml.common.input.parameter.regression.LogisticRegressionParams;
import org.opensearch.ml.common.input.parameter.sample.SampleAlgoParams;
import org.opensearch.ml.common.model.TextEmbeddingModelConfig;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskAction;
import org.opensearch.ml.common.transport.model.MLModelDeleteAction;
import org.opensearch.ml.common.transport.model.MLModelGetAction;
import org.opensearch.ml.common.transport.model.MLModelSearchAction;
import org.opensearch.ml.common.transport.model.forward.MLForwardAction;
import org.opensearch.ml.common.transport.model.load.MLLoadModelAction;
import org.opensearch.ml.common.transport.model.load.MLLoadModelOnNodeAction;
import org.opensearch.ml.common.transport.model.sync.MLSyncUpAction;
import org.opensearch.ml.common.transport.model.unload.MLUnloadModelAction;
import org.opensearch.ml.common.transport.model.upload.MLUploadModelAction;
import org.opensearch.ml.common.transport.model.upload_chunk.MLUploadModelChunkAction;
import org.opensearch.ml.common.transport.model.upload_chunk.MLUploadModelMetaAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.task.MLTaskDeleteAction;
import org.opensearch.ml.common.transport.task.MLTaskGetAction;
import org.opensearch.ml.common.transport.task.MLTaskSearchAction;
import org.opensearch.ml.common.transport.training.MLTrainingTaskAction;
import org.opensearch.ml.common.transport.trainpredict.MLTrainAndPredictionTaskAction;
import org.opensearch.ml.engine.MLEngine;
import org.opensearch.ml.engine.MLEngineClassLoader;
import org.opensearch.ml.engine.ModelHelper;
import org.opensearch.ml.engine.algorithms.anomalylocalization.AnomalyLocalizerImpl;
import org.opensearch.ml.engine.algorithms.sample.LocalSampleCalculator;
import org.opensearch.ml.indices.MLIndicesHandler;
import org.opensearch.ml.indices.MLInputDatasetHandler;
import org.opensearch.ml.model.MLModelManager;
import org.opensearch.ml.rest.RestMLDeleteModelAction;
import org.opensearch.ml.rest.RestMLDeleteTaskAction;
import org.opensearch.ml.rest.RestMLExecuteAction;
import org.opensearch.ml.rest.RestMLGetModelAction;
import org.opensearch.ml.rest.RestMLGetTaskAction;
import org.opensearch.ml.rest.RestMLLoadModelAction;
import org.opensearch.ml.rest.RestMLPredictionAction;
import org.opensearch.ml.rest.RestMLProfileAction;
import org.opensearch.ml.rest.RestMLSearchModelAction;
import org.opensearch.ml.rest.RestMLSearchTaskAction;
import org.opensearch.ml.rest.RestMLStatsAction;
import org.opensearch.ml.rest.RestMLTrainAndPredictAction;
import org.opensearch.ml.rest.RestMLTrainingAction;
import org.opensearch.ml.rest.RestMLUnloadModelAction;
import org.opensearch.ml.rest.RestMLUploadModelAction;
import org.opensearch.ml.rest.RestMLUploadModelChunkAction;
import org.opensearch.ml.rest.RestMLCreateModelMetaAction;
import org.opensearch.ml.settings.MLCommonsSettings;
import org.opensearch.ml.stats.MLClusterLevelStat;
import org.opensearch.ml.stats.MLNodeLevelStat;
import org.opensearch.ml.stats.MLStat;
import org.opensearch.ml.stats.MLStats;
import org.opensearch.ml.stats.suppliers.CounterSupplier;
import org.opensearch.ml.stats.suppliers.IndexStatusSupplier;
import org.opensearch.ml.task.MLExecuteTaskRunner;
import org.opensearch.ml.task.MLPredictTaskRunner;
import org.opensearch.ml.task.MLTaskDispatcher;
import org.opensearch.ml.task.MLTaskManager;
import org.opensearch.ml.task.MLTrainAndPredictTaskRunner;
import org.opensearch.ml.task.MLTrainingTaskRunner;
import org.opensearch.ml.utils.IndexUtils;
import org.opensearch.monitor.jvm.JvmService;
import org.opensearch.plugins.ActionPlugin;
import org.opensearch.plugins.Plugin;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.rest.RestController;
import org.opensearch.rest.RestHandler;
import org.opensearch.script.ScriptService;
import org.opensearch.threadpool.ExecutorBuilder;
import org.opensearch.threadpool.FixedExecutorBuilder;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.watcher.ResourceWatcherService;

import com.google.common.collect.ImmutableList;

public class MachineLearningPlugin extends Plugin implements ActionPlugin {
    public static final String TASK_THREAD_POOL = "OPENSEARCH_ML_TASK_THREAD_POOL";
    public static final String ML_BASE_URI = "/_plugins/_ml";

    private MLStats mlStats;
    private MLTaskManager mlTaskManager;
    private MLModelManager mlModelManager;
    private MLIndicesHandler mlIndicesHandler;
    private MLInputDatasetHandler mlInputDatasetHandler;
    private MLTrainingTaskRunner mlTrainingTaskRunner;
    private MLPredictTaskRunner mlPredictTaskRunner;
    private MLTrainAndPredictTaskRunner mlTrainAndPredictTaskRunner;
    private MLExecuteTaskRunner mlExecuteTaskRunner;
    private IndexUtils indexUtils;
    private ModelHelper modelHelper;
    private MLModelUploader mlModelUploader;
    private DiscoveryNodeHelper nodeHelper;
    private MLModelMetaUploader mlModelMetaUploader;
    private MLModelChunkUploader mlModelChunkUploader;

    private Client client;
    private ClusterService clusterService;
    private ThreadPool threadPool;
    private Set<String> indicesToListen;

    public static final String ML_ROLE_NAME = "ml";
    private NamedXContentRegistry xContentRegistry;

    @Override
    public List<ActionHandler<? extends ActionRequest, ? extends ActionResponse>> getActions() {
        return ImmutableList
            .of(
                new ActionHandler<>(MLStatsNodesAction.INSTANCE, MLStatsNodesTransportAction.class),
                new ActionHandler<>(MLExecuteTaskAction.INSTANCE, TransportExecuteTaskAction.class),
                new ActionHandler<>(MLPredictionTaskAction.INSTANCE, TransportPredictionTaskAction.class),
                new ActionHandler<>(MLTrainingTaskAction.INSTANCE, TransportTrainingTaskAction.class),
                new ActionHandler<>(MLTrainAndPredictionTaskAction.INSTANCE, TransportTrainAndPredictionTaskAction.class),
                new ActionHandler<>(MLModelGetAction.INSTANCE, GetModelTransportAction.class),
                new ActionHandler<>(MLModelDeleteAction.INSTANCE, DeleteModelTransportAction.class),
                new ActionHandler<>(MLModelSearchAction.INSTANCE, SearchModelTransportAction.class),
                new ActionHandler<>(MLTaskGetAction.INSTANCE, GetTaskTransportAction.class),
                new ActionHandler<>(MLTaskDeleteAction.INSTANCE, DeleteTaskTransportAction.class),
                new ActionHandler<>(MLTaskSearchAction.INSTANCE, SearchTaskTransportAction.class),
                new ActionHandler<>(MLProfileAction.INSTANCE, MLProfileTransportAction.class),
                new ActionHandler<>(MLUploadModelAction.INSTANCE, TransportUploadModelAction.class),
                new ActionHandler<>(MLLoadModelAction.INSTANCE, TransportLoadModelAction.class),
                new ActionHandler<>(MLLoadModelOnNodeAction.INSTANCE, TransportLoadModelOnNodeAction.class),
                new ActionHandler<>(MLUnloadModelAction.INSTANCE, TransportUnloadModelAction.class),
                new ActionHandler<>(MLUploadModelMetaAction.INSTANCE, TransportUploadModelMetaAction.class),
                new ActionHandler<>(MLUploadModelChunkAction.INSTANCE, TransportUploadModelChunkAction.class),
                new ActionHandler<>(MLForwardAction.INSTANCE, TransportForwardAction.class),
                new ActionHandler<>(MLSyncUpAction.INSTANCE, TransportSyncUpOnNodeAction.class)
            );
    }

    @Override
    public Collection<Object> createComponents(
        Client client,
        ClusterService clusterService,
        ThreadPool threadPool,
        ResourceWatcherService resourceWatcherService,
        ScriptService scriptService,
        NamedXContentRegistry xContentRegistry,
        Environment environment,
        NodeEnvironment nodeEnvironment,
        NamedWriteableRegistry namedWriteableRegistry,
        IndexNameExpressionResolver indexNameExpressionResolver,
        Supplier<RepositoriesService> repositoriesServiceSupplier
    ) {
        this.indexUtils = new IndexUtils(client, clusterService);
        this.client = client;
        this.threadPool = threadPool;
        this.clusterService = clusterService;
        this.xContentRegistry = xContentRegistry;
        Settings settings = environment.settings();
        MLEngine.setDjlCachePath(environment.dataFiles()[0]);

        JvmService jvmService = new JvmService(environment.settings());
        MLCircuitBreakerService mlCircuitBreakerService = new MLCircuitBreakerService(jvmService).init(environment.dataFiles()[0]);

        Map<Enum, MLStat<?>> stats = new ConcurrentHashMap<>();

        nodeHelper = new DiscoveryNodeHelper(clusterService);
        // cluster level stats
        stats.put(MLClusterLevelStat.ML_MODEL_INDEX_STATUS, new MLStat<>(true, new IndexStatusSupplier(indexUtils, ML_MODEL_INDEX)));
        stats.put(MLClusterLevelStat.ML_TASK_INDEX_STATUS, new MLStat<>(true, new IndexStatusSupplier(indexUtils, ML_TASK_INDEX)));
        stats.put(MLClusterLevelStat.ML_MODEL_COUNT, new MLStat<>(true, new CounterSupplier()));
        // node level stats
        stats.put(MLNodeLevelStat.ML_NODE_EXECUTING_TASK_COUNT, new MLStat<>(false, new CounterSupplier()));
        stats.put(MLNodeLevelStat.ML_NODE_TOTAL_REQUEST_COUNT, new MLStat<>(false, new CounterSupplier()));
        stats.put(MLNodeLevelStat.ML_NODE_TOTAL_FAILURE_COUNT, new MLStat<>(false, new CounterSupplier()));
        stats.put(MLNodeLevelStat.ML_NODE_TOTAL_MODEL_COUNT, new MLStat<>(false, new CounterSupplier()));
        stats.put(MLNodeLevelStat.ML_NODE_TOTAL_CIRCUIT_BREAKER_TRIGGER_COUNT, new MLStat<>(false, new CounterSupplier()));
        this.mlStats = new MLStats(stats);

        mlIndicesHandler = new MLIndicesHandler(clusterService, client);
        mlTaskManager = new MLTaskManager(client, mlIndicesHandler);
        modelHelper = new ModelHelper();
        mlModelManager = new MLModelManager(
            clusterService,
            client,
            threadPool,
            xContentRegistry,
            modelHelper,
            settings,
            mlStats,
            mlCircuitBreakerService
        );
        mlInputDatasetHandler = new MLInputDatasetHandler(client);
        mlModelUploader = new MLModelUploader(
            modelHelper,
            mlIndicesHandler,
            mlTaskManager,
            mlModelManager,
            threadPool,
            client,
            mlStats,
            mlCircuitBreakerService
        );
        mlModelMetaUploader = new MLModelMetaUploader(mlIndicesHandler, threadPool, client);
        mlModelChunkUploader = new MLModelChunkUploader(mlIndicesHandler, client, xContentRegistry);

        MLTaskDispatcher mlTaskDispatcher = new MLTaskDispatcher(clusterService, client, settings, nodeHelper);
        mlTrainingTaskRunner = new MLTrainingTaskRunner(
            threadPool,
            clusterService,
            client,
            mlTaskManager,
            mlStats,
            mlIndicesHandler,
            mlInputDatasetHandler,
            mlTaskDispatcher,
            mlCircuitBreakerService,
            nodeHelper
        );
        mlPredictTaskRunner = new MLPredictTaskRunner(
            threadPool,
            clusterService,
            client,
            mlTaskManager,
            mlStats,
            mlInputDatasetHandler,
            mlTaskDispatcher,
            mlCircuitBreakerService,
            xContentRegistry,
            mlModelManager,
            nodeHelper
        );
        mlTrainAndPredictTaskRunner = new MLTrainAndPredictTaskRunner(
            threadPool,
            clusterService,
            client,
            mlTaskManager,
            mlStats,
            mlInputDatasetHandler,
            mlTaskDispatcher,
            mlCircuitBreakerService,
            nodeHelper
        );
        mlExecuteTaskRunner = new MLExecuteTaskRunner(
            threadPool,
            clusterService,
            client,
            mlTaskManager,
            mlStats,
            mlInputDatasetHandler,
            mlTaskDispatcher,
            mlCircuitBreakerService,
            nodeHelper
        );

        // Register thread-safe ML objects here.
        LocalSampleCalculator localSampleCalculator = new LocalSampleCalculator(client, settings);
        MLEngineClassLoader.register(FunctionName.LOCAL_SAMPLE_CALCULATOR, localSampleCalculator);

        AnomalyLocalizerImpl anomalyLocalizer = new AnomalyLocalizerImpl(client, settings, clusterService, indexNameExpressionResolver);
        MLEngineClassLoader.register(FunctionName.ANOMALY_LOCALIZATION, anomalyLocalizer);

        MLSearchHandler mlSearchHandler = new MLSearchHandler(client, xContentRegistry);

        MLCommonsClusterEventListener mlCommonsClusterEventListener = new MLCommonsClusterEventListener(
            clusterService,
            mlModelManager,
            mlTaskManager
        );
        MLCommonsClusterManagerEventListener clusterManagerEventListener = new MLCommonsClusterManagerEventListener(
            clusterService,
            client,
            settings,
            threadPool,
            mlModelManager,
            mlTaskManager,
            nodeHelper
        );

        return ImmutableList
            .of(
                nodeHelper,
                mlStats,
                mlTaskManager,
                mlModelManager,
                mlIndicesHandler,
                mlInputDatasetHandler,
                mlTrainingTaskRunner,
                mlPredictTaskRunner,
                mlTrainAndPredictTaskRunner,
                mlExecuteTaskRunner,
                mlSearchHandler,
                mlTaskDispatcher,
                modelHelper,
                mlModelUploader,
                mlModelMetaUploader,
                mlModelChunkUploader,
                mlCommonsClusterEventListener,
                clusterManagerEventListener
            );
    }

    @Override
    public List<RestHandler> getRestHandlers(
        Settings settings,
        RestController restController,
        ClusterSettings clusterSettings,
        IndexScopedSettings indexScopedSettings,
        SettingsFilter settingsFilter,
        IndexNameExpressionResolver indexNameExpressionResolver,
        Supplier<DiscoveryNodes> nodesInCluster
    ) {
        RestMLStatsAction restMLStatsAction = new RestMLStatsAction(mlStats, clusterService, indexUtils);
        RestMLTrainingAction restMLTrainingAction = new RestMLTrainingAction();
        RestMLTrainAndPredictAction restMLTrainAndPredictAction = new RestMLTrainAndPredictAction();
        RestMLPredictionAction restMLPredictionAction = new RestMLPredictionAction();
        RestMLExecuteAction restMLExecuteAction = new RestMLExecuteAction();
        RestMLGetModelAction restMLGetModelAction = new RestMLGetModelAction();
        RestMLDeleteModelAction restMLDeleteModelAction = new RestMLDeleteModelAction();
        RestMLSearchModelAction restMLSearchModelAction = new RestMLSearchModelAction();
        RestMLGetTaskAction restMLGetTaskAction = new RestMLGetTaskAction();
        RestMLDeleteTaskAction restMLDeleteTaskAction = new RestMLDeleteTaskAction();
        RestMLSearchTaskAction restMLSearchTaskAction = new RestMLSearchTaskAction();
        RestMLProfileAction restMLProfileAction = new RestMLProfileAction(clusterService);
        RestMLUploadModelAction restMLUploadModelAction = new RestMLUploadModelAction();
        RestMLLoadModelAction restMLLoadModelAction = new RestMLLoadModelAction();
        RestMLUnloadModelAction restMLUnloadModelAction = new RestMLUnloadModelAction(clusterService);
        RestMLCreateModelMetaAction restMLUploadModelMetaAction = new RestMLCreateModelMetaAction();
        RestMLUploadModelChunkAction restMLCustomModelUploadChunkAction = new RestMLUploadModelChunkAction();

        return ImmutableList
            .of(
                restMLStatsAction,
                restMLTrainingAction,
                restMLPredictionAction,
                restMLExecuteAction,
                restMLTrainAndPredictAction,
                restMLGetModelAction,
                restMLDeleteModelAction,
                restMLSearchModelAction,
                restMLGetTaskAction,
                restMLDeleteTaskAction,
                restMLSearchTaskAction,
                restMLProfileAction,
                restMLUploadModelAction,
                restMLLoadModelAction,
                restMLUnloadModelAction,
                restMLUploadModelMetaAction,
                restMLCustomModelUploadChunkAction
            );
    }

    @Override
    public List<ExecutorBuilder<?>> getExecutorBuilders(Settings settings) {
        FixedExecutorBuilder ml = new FixedExecutorBuilder(settings, TASK_THREAD_POOL, 4, 4, "ml.task_thread_pool", false);

        return Collections.singletonList(ml);
    }

    @Override
    public List<NamedXContentRegistry.Entry> getNamedXContent() {
        return ImmutableList
            .of(
                KMeansParams.XCONTENT_REGISTRY,
                LinearRegressionParams.XCONTENT_REGISTRY,
                AnomalyDetectionLibSVMParams.XCONTENT_REGISTRY,
                SampleAlgoParams.XCONTENT_REGISTRY,
                FitRCFParams.XCONTENT_REGISTRY,
                BatchRCFParams.XCONTENT_REGISTRY,
                LocalSampleCalculatorInput.XCONTENT_REGISTRY,
                AnomalyLocalizationInput.XCONTENT_REGISTRY_ENTRY,
                RCFSummarizeParams.XCONTENT_REGISTRY,
                LogisticRegressionParams.XCONTENT_REGISTRY,
                TextEmbeddingModelConfig.XCONTENT_REGISTRY
            );
    }

    @Override
    public List<Setting<?>> getSettings() {
        List<Setting<?>> settings = ImmutableList
            .of(
                MLCommonsSettings.ML_COMMONS_TASK_DISPATCH_POLICY,
                MLCommonsSettings.ML_COMMONS_MAX_MODELS_PER_NODE,
                MLCommonsSettings.ML_COMMONS_ONLY_RUN_ON_ML_NODE,
                MLCommonsSettings.ML_COMMONS_SYNC_UP_JOB_INTERVAL_IN_SECONDS,
                MLCommonsSettings.ML_COMMONS_MONITORING_REQUEST_COUNT
            );
        return settings;
    }
}
