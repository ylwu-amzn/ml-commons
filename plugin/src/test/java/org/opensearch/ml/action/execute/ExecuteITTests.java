package org.opensearch.ml.action.execute;

import com.google.common.collect.ImmutableList;
import org.junit.Before;
import org.junit.Rule;
import org.junit.rules.ExpectedException;
import org.opensearch.action.ActionRequest;
import org.opensearch.ml.action.MLCommonsIntegTestCase;
import org.opensearch.ml.common.parameter.Input;
import org.opensearch.ml.common.parameter.LocalSampleCalculatorInput;
import org.opensearch.ml.common.parameter.MLInput;
import org.opensearch.ml.common.parameter.MLModel;
import org.opensearch.ml.common.parameter.Output;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskAction;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskRequest;
import org.opensearch.ml.common.transport.execute.MLExecuteTaskResponse;
import org.opensearch.ml.common.transport.task.MLTaskGetAction;
import org.opensearch.ml.common.transport.task.MLTaskGetResponse;
import org.opensearch.test.OpenSearchIntegTestCase;

@OpenSearchIntegTestCase.ClusterScope(scope = OpenSearchIntegTestCase.Scope.SUITE, numDataNodes = 3)
public class ExecuteITTests extends MLCommonsIntegTestCase {

    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();

    @Before
    public void setUp() throws Exception {
        super.setUp();
    }

    public void testExecuteLocalSampleCalculator() {
        MLInput input = new LocalSampleCalculatorInput("sum", ImmutableList.of(1.0, 2.0, 3.0));
        MLExecuteTaskRequest request = new MLExecuteTaskRequest(input);
        MLExecuteTaskResponse executeTaskResponse = client().execute(MLExecuteTaskAction.INSTANCE, request).actionGet(5000);
        Output output = executeTaskResponse.getOutput();
    }
}
