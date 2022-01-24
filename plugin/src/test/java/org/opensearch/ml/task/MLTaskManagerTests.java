/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 *
 */

package org.opensearch.ml.task;

import static org.mockito.Mockito.*;

import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.mockito.ArgumentCaptor;
import org.opensearch.action.ActionListener;
import org.opensearch.action.update.UpdateResponse;
import org.opensearch.client.Client;
import org.opensearch.ml.common.dataframe.ColumnMeta;
import org.opensearch.ml.common.dataframe.ColumnType;
import org.opensearch.ml.common.dataframe.ColumnValue;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.dataframe.DefaultDataFrame;
import org.opensearch.ml.common.dataframe.DoubleValue;
import org.opensearch.ml.common.dataframe.Row;
import org.opensearch.ml.common.parameter.MLPredictionOutput;
import org.opensearch.ml.indices.MLIndicesHandler;
import org.opensearch.ml.model.MLTask;
import org.opensearch.ml.model.MLTaskState;
import org.opensearch.ml.model.MLTaskType;

import com.google.common.collect.ImmutableMap;

public class MLTaskManagerTests {
    MLTaskManager mlTaskManager;
    MLTask mlTask;
    Client client;
    MLIndicesHandler mlIndicesHandler;

    @Rule
    public ExpectedException expectedEx = ExpectedException.none();

    @Before
    public void setup() {
        this.client = mock(Client.class);
        this.mlIndicesHandler = mock(MLIndicesHandler.class);
        this.mlTaskManager = spy(new MLTaskManager(client, mlIndicesHandler));
        this.mlTask = MLTask
            .builder()
            .taskId("task id")
            .taskType(MLTaskType.PREDICTION)
            .createTime(Instant.now())
            .state(MLTaskState.CREATED)
            .build();
    }

    @Test
    public void testAdd() {
        expectedEx.expect(IllegalArgumentException.class);
        expectedEx.expectMessage("Duplicate taskId");
        mlTaskManager.add(mlTask);
        Assert.assertTrue(mlTaskManager.contains(mlTask.getTaskId()));
        mlTaskManager.add(mlTask);
    }

    @Test
    public void testAddIfAbsent() {
        mlTaskManager.addIfAbsent(mlTask);
        Assert.assertTrue(mlTaskManager.contains(mlTask.getTaskId()));
        mlTaskManager.addIfAbsent(mlTask);
        Assert.assertTrue(mlTaskManager.contains(mlTask.getTaskId()));
    }

    @Test
    public void testUpdateTaskState() {
        expectedEx.expect(RuntimeException.class);
        expectedEx.expectMessage("Task not found");
        mlTaskManager.add(mlTask);
        mlTaskManager.updateTaskState(mlTask.getTaskId(), MLTaskState.RUNNING, true);
        Assert.assertSame(mlTaskManager.get(mlTask.getTaskId()).getState(), MLTaskState.RUNNING);
        mlTaskManager.updateTaskState("not exist", MLTaskState.RUNNING, true);
    }

    @Test
    public void testUpdateTaskError() {
        expectedEx.expect(RuntimeException.class);
        expectedEx.expectMessage("Task not found");
        mlTaskManager.add(mlTask);
        mlTaskManager.updateTaskError(mlTask.getTaskId(), "error message", true);
        Assert.assertEquals("error message", mlTaskManager.get(mlTask.getTaskId()).getError());
        mlTaskManager.updateTaskError("not exist", "error message", true);
    }

    @Test
    public void testUpdateTaskStateAndError() {
        mlTaskManager.add(mlTask);
        String error = "error message";
        mlTaskManager.updateTaskStateAndError(mlTask.getTaskId(), MLTaskState.FAILED, error, true);
        ArgumentCaptor<Map> updatedFields = ArgumentCaptor.forClass(Map.class);
        ArgumentCaptor<Long> timeoutValue = ArgumentCaptor.forClass(Long.class);
        verify(mlTaskManager).updateMLTask(eq(mlTask.getTaskId()), updatedFields.capture(), timeoutValue.capture());
        Map map = updatedFields.getValue();
        Assert.assertEquals(2, map.size());
        Assert.assertEquals(MLTaskState.FAILED.name(), map.get("state"));
        Assert.assertEquals(error, map.get("error"));
        Long value = timeoutValue.getValue();
        Assert.assertEquals(0, value.longValue());
    }

    @Test
    public void testUpdateMLTaskWithNullOrEmptyMap() {
        mlTaskManager.add(mlTask);
        ActionListener<UpdateResponse> listener = mock(ActionListener.class);
        mlTaskManager.updateMLTask(mlTask.getTaskId(), null, listener, 0);
        verify(client, never()).index(any());
        verify(listener, times(1)).onFailure(any());

        mlTaskManager.updateMLTask(mlTask.getTaskId(), new HashMap<>(), listener, 0);
        verify(client, never()).index(any());
        verify(listener, times(2)).onFailure(any());
    }

    @Test
    public void testRemove() {
        mlTaskManager.add(mlTask);
        Assert.assertTrue(mlTaskManager.contains(mlTask.getTaskId()));
        mlTaskManager.remove(mlTask.getTaskId());
        Assert.assertFalse(mlTaskManager.contains(mlTask.getTaskId()));
    }

    @Test
    public void testGetRunningTaskCount() {
        MLTask task1 = MLTask.builder().taskId("1").state(MLTaskState.CREATED).build();
        MLTask task2 = MLTask.builder().taskId("2").state(MLTaskState.RUNNING).build();
        MLTask task3 = MLTask.builder().taskId("3").state(MLTaskState.FAILED).build();
        MLTask task4 = MLTask.builder().taskId("4").state(MLTaskState.COMPLETED).build();
        mlTaskManager.add(task1);
        mlTaskManager.add(task2);
        mlTaskManager.add(task3);
        mlTaskManager.add(task4);
        Assert.assertEquals(mlTaskManager.getRunningTaskCount(), 1);
    }

    @Test
    public void testClear() {
        MLTask task1 = MLTask.builder().taskId("1").state(MLTaskState.CREATED).build();
        MLTask task2 = MLTask.builder().taskId("2").state(MLTaskState.RUNNING).build();
        MLTask task3 = MLTask.builder().taskId("3").state(MLTaskState.FAILED).build();
        MLTask task4 = MLTask.builder().taskId("4").state(MLTaskState.COMPLETED).build();
        mlTaskManager.add(task1);
        mlTaskManager.add(task2);
        mlTaskManager.add(task3);
        mlTaskManager.add(task4);
        mlTaskManager.clear();
        Assert.assertFalse(mlTaskManager.contains(task1.getTaskId()));
        Assert.assertFalse(mlTaskManager.contains(task2.getTaskId()));
        Assert.assertFalse(mlTaskManager.contains(task3.getTaskId()));
        Assert.assertFalse(mlTaskManager.contains(task4.getTaskId()));
    }

    @Test
    public void testCreateMlTask_InitIndexReturnFalse() {
        doAnswer(invocation -> {
            ActionListener<Boolean> listener = invocation.getArgument(0);
            listener.onResponse(false);
            return null;
        }).when(mlIndicesHandler).initMLTaskIndex(any(ActionListener.class));

        ActionListener listener = mock(ActionListener.class);
        mlTaskManager.createMLTask(mlTask, listener);
        verify(listener).onFailure(any());
    }

    @Test
    public void testCreateMlTask_IndexException() {
        doAnswer(invocation -> {
            ActionListener<Boolean> listener = invocation.getArgument(0);
            listener.onResponse(true);
            return null;
        }).when(mlIndicesHandler).initMLTaskIndex(any(ActionListener.class));

        doThrow(new RuntimeException("test")).when(client).index(any(), any());
        ActionListener listener = mock(ActionListener.class);
        mlTaskManager.createMLTask(mlTask, listener);
        verify(listener).onFailure(any());
    }

    @Test
    public void testA() {
        ColumnMeta[] columnMetas = new ColumnMeta[] { new ColumnMeta("a", ColumnType.DOUBLE) };
        List<Row> rows = new ArrayList<>();
        rows.add(new Row(new ColumnValue[] { new DoubleValue(1.0) }));
        rows.add(new Row(new ColumnValue[] { new DoubleValue(10.0) }));
        DataFrame dataFrame = new DefaultDataFrame(columnMetas, rows);
        MLPredictionOutput predictionResult = new MLPredictionOutput("taskId", "status", dataFrame);
        System.out.println("++++++++++++++++++++++++++++++++++++++++");
        System.out.println(predictionResult.getPredictionResult());
        System.out.println("++++++++++++++++++++++++++++++++++++++++");
    }

    @Test
    public void testBb() {
        ColumnMeta[] columnMetas = new ColumnMeta[] { new ColumnMeta("a", ColumnType.DOUBLE) };
        List<Row> rows = new ArrayList<>();
        rows.add(new Row(new ColumnValue[] { new DoubleValue(1.0) }));
        rows.add(new Row(new ColumnValue[] { new DoubleValue(10.0) }));
        DataFrame inputDataFrame = new DefaultDataFrame(columnMetas, rows);

        ColumnMeta[] columnMetas2 = new ColumnMeta[] { new ColumnMeta("b", ColumnType.DOUBLE) };
        List<Row> rows2 = new ArrayList<>();
        rows2.add(new Row(new ColumnValue[] { new DoubleValue(2.0) }));
        rows2.add(new Row(new ColumnValue[] { new DoubleValue(20.0) }));
        DataFrame inputDataFrame2 = new DefaultDataFrame(columnMetas2, rows2);

        // inputDataFrame.appendColumn(new ColumnMeta("testt", ColumnType.DOUBLE), new Object[]{2.0, 4.0});
        System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        System.out.println(inputDataFrame);
        System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        System.out.println(inputDataFrame2);
        System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");

        Iterator<Row> iterator1 = inputDataFrame.iterator();
        Iterator<Row> iterator2 = inputDataFrame2.iterator();
        Iterator<String> iterator = new Iterator<String>() {
            @Override
            public boolean hasNext() {
                return iterator1.hasNext();
            }

            @Override
            public String next() {
                StringBuilder sb = new StringBuilder();
                sb.append(convertRowIntoExprValue(inputDataFrame.columnMetas(), iterator1.next()));
                sb.append(convertRowIntoExprValue(inputDataFrame2.columnMetas(), iterator2.next()));
                return sb.toString();
            }
        };

        while (iterator.hasNext()) {
            String next = iterator.next();
            System.out.println(next);
        }
    }

    private java.lang.String convertRowIntoExprValue(ColumnMeta[] columnMetas, Row row) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < columnMetas.length; i++) {
            ColumnValue columnValue = row.getValue(i);
            String name = columnMetas[i].getName();
            sb.append(name).append(":");
            switch (columnValue.columnType()) {
                case INTEGER:
                    sb.append(columnValue.intValue()).append("; ");
                    break;
                case DOUBLE:
                    sb.append(columnValue.doubleValue()).append("; ");
                    break;
                case STRING:
                    sb.append(columnValue.stringValue()).append("; ");
                    break;
                default:
                    break;
            }
        }
        return sb.toString();
    }

    @Test
    public void testCc() {
        ImmutableMap.Builder<String, String> builder = new ImmutableMap.Builder<>();
        builder.putAll(ImmutableMap.of("a", "1", "b", "2"));
        builder.putAll(ImmutableMap.of("c", "100"));
        ImmutableMap<String, String> build = builder.build();
        System.out.println("----------------------------------------");
        System.out.println(build);
        System.out.println("----------------------------------------");
    }
}
