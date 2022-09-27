package org.opensearch.ml.common.model;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.junit.Assert.assertEquals;

public class MLModelFormatTests {

    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();

    @Test
    public void from() {
        MLModelFormat modelFormat = MLModelFormat.from("TORCH_SCRIPT");
        assertEquals(MLModelFormat.TORCH_SCRIPT, modelFormat);
    }

    @Test
    public void from_wrongValue() {
        exceptionRule.expect(IllegalArgumentException.class);
        exceptionRule.expectMessage("Wrong model format");
        MLModelFormat.from("test_wrong_value");
    }
}
