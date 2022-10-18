/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.task;

import static org.opensearch.ml.common.CommonValue.ML_TASK_INDEX;
import static org.opensearch.ml.common.MLTask.LAST_UPDATE_TIME_FIELD;

import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

import lombok.extern.log4j.Log4j2;

import org.opensearch.action.ActionListener;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.index.IndexResponse;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.action.update.UpdateRequest;
import org.opensearch.action.update.UpdateResponse;
import org.opensearch.client.Client;
import org.opensearch.common.Strings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.ToXContent;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.ml.common.MLTask;
import org.opensearch.ml.common.MLTaskState;
import org.opensearch.ml.common.MLTaskType;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.exception.MLResourceNotFoundException;
import org.opensearch.ml.indices.MLIndicesHandler;
import org.opensearch.rest.RestStatus;

/**
 * MLTaskManager is responsible for managing MLTask.
 */
@Log4j2
public class MLTaskManager {
    private final Map<String, MLTaskCache> taskCaches;
    // TODO: make this value configurable as cluster setting
    public final static int MAX_ML_TASK_PER_NODE = 10;
    private final Client client;
    private final MLIndicesHandler mlIndicesHandler;

    /**
     * Constructor to create ML task manager.
     *
     * @param client client
     * @param mlIndicesHandler ML indices handler
     */
    public MLTaskManager(Client client, MLIndicesHandler mlIndicesHandler) {
        this.client = client;
        this.mlIndicesHandler = mlIndicesHandler;
        taskCaches = new ConcurrentHashMap<>();
    }

    /**
     * Put ML task into cache.
     * If ML task is already in cache, will throw {@link IllegalArgumentException}
     *
     * @param mlTask ML task
     */
    public synchronized void add(MLTask mlTask) {
        // todo: once circuit break is in place, we need to add those checks
        // to make sure we have some limitation while adding new tasks.
        add(mlTask, null);
    }

    public synchronized void add(MLTask mlTask, List<String> workerNodes) {
        mlTask.setState(MLTaskState.RUNNING);
        String taskId = mlTask.getTaskId();
        if (contains(taskId)) {
            throw new IllegalArgumentException("Duplicate taskId");
        }
        taskCaches.put(taskId, new MLTaskCache(mlTask, workerNodes));
        log.debug("add ML task to cache " + taskId);
    }

    /**
     * Update ML task state
     * @param taskId task id
     * @param state MLTaskState
     * @param isAsyncTask is async task or not
     */
    public synchronized void updateTaskState(String taskId, MLTaskState state, boolean isAsyncTask) {
        updateTaskStateAndError(taskId, state, null, isAsyncTask);
    }

    /**
     * Update task error
     * @param taskId task id
     * @param error error message
     * @param isAsyncTask is async task
     */
    public synchronized void updateTaskError(String taskId, String error, boolean isAsyncTask) {
        updateTaskStateAndError(taskId, null, error, isAsyncTask);
    }

    public synchronized void updateTaskStateAndError(String taskId, MLTaskState state, String error, boolean isAsyncTask) {
        if (!contains(taskId)) {
            throw new IllegalArgumentException("Task not found");
        }
        MLTask task = getMLTask(taskId);
        task.setState(state);
        task.setError(error);
        if (isAsyncTask) {
            Map<String, Object> updatedFields = new HashMap<>();
            if (state != null) {
                updatedFields.put(MLTask.STATE_FIELD, state.name());
            }
            if (error != null) {
                updatedFields.put(MLTask.ERROR_FIELD, error);
            }
            updateMLTask(taskId, updatedFields, 0);
        }
    }

    /**
     * Check if task exists in cache.
     *
     * @param taskId task id
     * @return true if task exists in cache; otherwise, return false.
     */
    public boolean contains(String taskId) {
        return taskCaches.containsKey(taskId);
    }

    /**
     * Remove task from cache.
     *
     * @param taskId ML task id
     */
    public void remove(String taskId) {
        if (contains(taskId)) {
            taskCaches.remove(taskId);
            log.debug("remove ML task from cache " + taskId);
        }
    }

    /**
     * Get task from cache.
     *
     * @param taskId ML task id
     * @return ML task
     */
    public MLTask getMLTask(String taskId) {
        if (contains(taskId)) {
            return taskCaches.get(taskId).getMlTask();
        }
        return null;
    }

    public MLTaskCache getMLTaskCache(String taskId) {
        if (contains(taskId)) {
            return taskCaches.get(taskId);
        }
        return null;
    }

    public List<String> getWorkNodes(String taskId) {
        if (taskCaches.containsKey(taskId)) {
            return taskCaches.get(taskId).getWorkerNodes();
        }
        return null;
    }

    public void addNodeError(String taskId, String workerNodeId, String error) {
        if (taskCaches.containsKey(taskId)) {
            taskCaches.get(taskId).addError(workerNodeId, error);
        }
    }

    /**
     * Get all taskIds from cache
     * @return an array of all the keys in the taskCaches
     */
    public String[] getAllTaskIds() {
        return Strings.toStringArray(taskCaches.keySet());
    }

    /**
     * Get running task count in cache.
     *
     * @return running task count
     */
    public int getRunningTaskCount() {
        int res = 0;
        for (Map.Entry<String, MLTaskCache> entry : taskCaches.entrySet()) {
            MLTask mlTask = entry.getValue().getMlTask();
            if (mlTask.getState() != null && mlTask.getState() == MLTaskState.RUNNING) {
                res++;
            }
        }
        return res;
    }

    public int getRunningTaskCount(MLTaskType taskType) {
        int res = 0;
        for (Map.Entry<String, MLTaskCache> entry : taskCaches.entrySet()) {
            MLTask mlTask = entry.getValue().getMlTask();
            if (mlTask.getState() != null && mlTask.getState() == MLTaskState.RUNNING && mlTask.getTaskType() == taskType) {
                res++;
            }
        }
        return res;
    }

    /**
     * Clear all tasks.
     */
    public void clear() {
        taskCaches.clear();
    }

    /**
     * Create ML task. Will init ML task index first if absent.
     * @param mlTask ML task
     * @param listener action listener
     */
    public void createMLTask(MLTask mlTask, ActionListener<IndexResponse> listener) {
        mlIndicesHandler.initMLTaskIndex(ActionListener.wrap(indexCreated -> {
            if (!indexCreated) {
                listener.onFailure(new RuntimeException("No response to create ML task index"));
                return;
            }
            IndexRequest request = new IndexRequest(ML_TASK_INDEX);
            try (
                XContentBuilder builder = XContentFactory.jsonBuilder();
                ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()
            ) {
                request.source(mlTask.toXContent(builder, ToXContent.EMPTY_PARAMS)).setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
                client.index(request, ActionListener.runBefore(listener, () -> context.restore()));
            } catch (Exception e) {
                log.error("Failed to create AD task for " + mlTask.getFunctionName() + ", " + mlTask.getTaskType(), e);
                listener.onFailure(e);
            }
        }, e -> {
            log.error("Failed to create ML index", e);
            listener.onFailure(e);
        }));
    }

    /**
     * Update ML task with default listener.
     * @param taskId task id
     * @param updatedFields updated field and values
     * @param timeoutInMillis time out waiting for updating task semaphore, zero or negative means don't wait at all
     */
    public void updateMLTask(String taskId, Map<String, Object> updatedFields, long timeoutInMillis) {
        updateMLTask(taskId, updatedFields, ActionListener.wrap(response -> {
            if (response.status() == RestStatus.OK) {
                log.debug("Updated ML task successfully: {}, task id: {}", response.status(), taskId);
            } else {
                log.error("Failed to update ML task {}, status: {}", taskId, response.status());
            }
        }, e -> { log.error("Failed to update ML task: " + taskId, e); }), timeoutInMillis);
    }

    /**
     * Update ML task.
     * @param taskId task id
     * @param updatedFields updated field and values
     * @param listener action listener
     * @param timeoutInMillis time out waiting for updating task semaphore, zero or negative means don't wait at all
     */
    public void updateMLTask(
        String taskId,
        Map<String, Object> updatedFields,
        ActionListener<UpdateResponse> listener,
        long timeoutInMillis
    ) {
        if (!taskCaches.containsKey(taskId)) {
            listener.onFailure(new MLResourceNotFoundException("Can't find task"));
            return;
        }
        Semaphore semaphore = taskCaches.get(taskId).getUpdateTaskIndexSemaphore();
        try {
            if (semaphore != null && !semaphore.tryAcquire(timeoutInMillis, TimeUnit.MILLISECONDS)) {
                listener.onFailure(new MLException("Other updating request not finished yet"));
                return;
            }
        } catch (InterruptedException e) {
            log.error("Failed to acquire semaphore for ML task " + taskId, e);
            listener.onFailure(e);
            return; // return directly if can't get semaphore
        }
        try {
            if (updatedFields == null || updatedFields.size() == 0) {
                listener.onFailure(new IllegalArgumentException("Updated fields is null or empty"));
                return;
            }
            UpdateRequest updateRequest = new UpdateRequest(ML_TASK_INDEX, taskId);
            Map<String, Object> updatedContent = new HashMap<>();
            updatedContent.putAll(updatedFields);
            updatedContent.put(LAST_UPDATE_TIME_FIELD, Instant.now().toEpochMilli());
            updateRequest.doc(updatedContent);
            updateRequest.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            ActionListener<UpdateResponse> actionListener = semaphore == null
                ? listener
                : ActionListener.runAfter(listener, () -> semaphore.release());
            try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                client.update(updateRequest, ActionListener.runBefore(actionListener, () -> context.restore()));
            } catch (Exception e) {
                actionListener.onFailure(e);
            }
        } catch (Exception e) {
            semaphore.release();
            log.error("Failed to update ML task " + taskId, e);
            listener.onFailure(e);
        }
    }

    public void updateMLTaskDirectly(String taskId, Map<String, Object> updatedFields) {
        updateMLTaskDirectly(taskId, updatedFields, ActionListener.wrap(r -> {
            log.debug("updated ML task directly: {}", taskId);
        }, e-> {
            log.error("Failed to update ML task " + taskId, e);
        }));
    }
    public void updateMLTaskDirectly(String taskId, Map<String, Object> updatedFields, ActionListener<UpdateResponse> listener) {
        try {
            if (updatedFields == null || updatedFields.size() == 0) {
                listener.onFailure(new IllegalArgumentException("Updated fields is null or empty"));
                return;
            }
            UpdateRequest updateRequest = new UpdateRequest(ML_TASK_INDEX, taskId);
            Map<String, Object> updatedContent = new HashMap<>();
            updatedContent.putAll(updatedFields);
            updatedContent.put(LAST_UPDATE_TIME_FIELD, Instant.now().toEpochMilli());
            updateRequest.doc(updatedContent);
            updateRequest.setRefreshPolicy(WriteRequest.RefreshPolicy.IMMEDIATE);
            try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
                client.update(updateRequest, ActionListener.runBefore(listener, () -> context.restore()));
            } catch (Exception e) {
                listener.onFailure(e);
            }
        } catch (Exception e) {
            log.error("Failed to update ML task " + taskId, e);
            listener.onFailure(e);
        }
    }

    public boolean containsModel(String modelId) {
        for (Map.Entry<String, MLTaskCache> entry : taskCaches.entrySet()) {
            if (modelId.equals(entry.getValue().mlTask.getModelId())) {
                return true;
            }
        }
        return false;
    }
}
