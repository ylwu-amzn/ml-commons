package org.opensearch.ml.common.model;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.junit.Assert.assertEquals;

public class MLModelStateTests {

    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();

    @Test
    public void from() {
        MLModelState modelState = MLModelState.from("UPLOADED");
        assertEquals(MLModelState.UPLOADED, modelState);
    }

    @Test
    public void from_wrongValue() {
        exceptionRule.expect(IllegalArgumentException.class);
        exceptionRule.expectMessage("Wrong model format");
        MLModelState.from("test_wrong_value");
    }
}
