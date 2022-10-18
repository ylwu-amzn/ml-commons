/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.model;

import com.google.common.collect.ImmutableMap;
import lombok.extern.log4j.Log4j2;
import org.opensearch.action.ActionListener;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.action.update.UpdateRequest;
import org.opensearch.action.update.UpdateResponse;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.breaker.MLCircuitBreakerService;
import org.opensearch.ml.engine.ModelHelper;
import org.opensearch.ml.stats.MLStats;
import org.opensearch.threadpool.ThreadPool;

import static org.opensearch.ml.common.CommonValue.ML_MODEL_INDEX;
import static org.opensearch.ml.settings.MLCommonsSettings.ML_COMMONS_MAX_MODELS_PER_NODE;

@Log4j2
public class MLModelManager {

    private final Client client;
    private ThreadPool threadPool;
    private NamedXContentRegistry xContentRegistry;
    private ModelHelper modelHelper;

    private final MLModelCache modelCache;
    private volatile Integer maxModelPerNode;

    private final MLStats mlStats;
    protected final MLCircuitBreakerService mlCircuitBreakerService;

    public MLModelManager(
        ClusterService clusterService,
        Client client,
        ThreadPool threadPool,
        NamedXContentRegistry xContentRegistry,
        ModelHelper modelHelper,
        Settings settings,
        MLStats mlStats,
        MLCircuitBreakerService mlCircuitBreakerService
    ) {
        this.client = client;
        this.threadPool = threadPool;
        this.xContentRegistry = xContentRegistry;
        this.modelHelper = modelHelper;
        this.modelCache = new MLModelCache(clusterService, settings);
        this.mlStats = mlStats;
        this.maxModelPerNode = ML_COMMONS_MAX_MODELS_PER_NODE.get(settings);
        this.mlCircuitBreakerService = mlCircuitBreakerService;
        clusterService.getClusterSettings().addSettingsUpdateConsumer(ML_COMMONS_MAX_MODELS_PER_NODE, it -> maxModelPerNode = it);
    }


    public void updateModel(String modelId, ImmutableMap<String, Object> updatedFields, ActionListener<UpdateResponse> listener) {
        try {
            if (updatedFields == null || updatedFields.size() == 0) {
                listener.onFailure(new IllegalArgumentException("Updated fields is null or empty"));
                return;
            }
            UpdateRequest updateRequest = new UpdateRequest(ML_MODEL_INDEX, modelId);
            updateRequest.doc(updatedFields);
            updateRequest.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                client.update(updateRequest, ActionListener.runBefore(listener, () -> context.restore()));
            } catch (Exception e) {
                listener.onFailure(e);
            }
        } catch (Exception e) {
            log.error("Failed to update ML model " + modelId, e);
            listener.onFailure(e);
        }
    }

    public String getModelChunkId(String modelId, Integer chunkNumber) {
        return modelId + "_" + chunkNumber;
    }


}
