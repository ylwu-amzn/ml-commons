/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.ml.common.controller;


import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.search.SearchModule;

public class MLModelControllerTest {
    private  MLRateLimiter rateLimiter;

    private MLModelController modelController;

    private MLModelController modelControllerNull;

    private final String expectedInputStr = "{\"model_id\":\"testModelId\",\"user_rate_limiter_config\":" +
            "{\"testUser\":{\"rate_limit_number\":\"1\",\"rate_limit_unit\":\"MILLISECONDS\"}}}";

    @Rule
    public ExpectedException exceptionRule = ExpectedException.none();

    @Before
    public void setUp() throws Exception {
        rateLimiter = MLRateLimiter.builder()
                .limit(1.0)
                .unit(TimeUnit.MILLISECONDS)
                .build();

        modelControllerNull = MLModelController.builder()
                .modelId("testModelId").build();

        modelController = MLModelControllerGenerator("testUser", rateLimiter);

    }

    @Test
    public void readInputStreamSuccess() throws IOException {
        readInputStream(modelController, parsedInput -> {
            assertEquals("testModelId", parsedInput.getModelId());
            assertEquals(modelController.getUserRateLimiterConfig().get("testUser").getLimit(),
                    parsedInput.getUserRateLimiterConfig().get("testUser").getLimit());
        });
    }

    @Test
    public void readInputStreamSuccessWithNullFields() throws IOException {
        modelController.setUserRateLimiterConfig(null);
        readInputStream(modelController, parsedInput -> {
            assertNull(parsedInput.getUserRateLimiterConfig());
        });
    }

    @Test
    public void testToXContent() throws Exception {
        String jsonStr = serializationWithToXContent(modelController);
        assertEquals(expectedInputStr, jsonStr);
    }


    @Test
    public void testToXContentIncomplete() throws Exception {
        final String expectedIncompleteInputStr =
                "{\"model_id\":\"testModelId\"}";
        String jsonStr = serializationWithToXContent(modelControllerNull);
        assertEquals(expectedIncompleteInputStr, jsonStr);
    }

    @Test
    public void testToXContentWithNullMLRateLimiterInUserRateLimiterConfig() throws Exception {
        // Notice that MLModelController will throw an exception if it parses this output string, check parseWithNullMLRateLimiterInUserRateLimiterConfigFieldWithException test below.
        final String expectedOutputStrWithNullField =
                "{\"model_id\":\"testModelId\",\"user_rate_limiter_config\":{\"testUser\":null}}";
        MLModelController modelControllerWithTestUserAndEmptyRateLimiter = MLModelController.builder()
                .modelId("testModelId")
                .userRateLimiterConfig(new HashMap<>(){{put("testUser", null);}})
                .build();
        String jsonStr = serializationWithToXContent(modelControllerWithTestUserAndEmptyRateLimiter);
        assertEquals(expectedOutputStrWithNullField, jsonStr);
    }

    @Test
    public void parseSuccess() throws Exception {
        testParseFromJsonString(expectedInputStr, parsedInput -> assertEquals("testModelId", parsedInput.getModelId()));
    }

    @Test
    // Notice that this won't throw an IllegalStateException, which is pretty different from usual
    public void parseWithoutUserRateLimiterConfigFieldWithNoException() throws Exception {
        final String expectedIncompleteInputStr = "{\"model_id\":\"testModelId\"}";
        final String expectedOutputStr = "{\"model_id\":\"testModelId\",\"user_rate_limiter_config\":{}}";

        testParseFromJsonString(expectedIncompleteInputStr, parsedInput -> {
            try {
                assertEquals(expectedOutputStr, serializationWithToXContent(parsedInput));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    @Test
    // Notice that this won't throw an IllegalStateException, which is pretty different from usual
    public void parseWithNullUserRateLimiterConfigFieldWithNoException() throws Exception {
        final String expectedInputStrWithNullField = "{\"model_id\":\"testModelId\",\"user_rate_limiter_config\":null}";
        final String expectedOutputStr = "{\"model_id\":\"testModelId\",\"user_rate_limiter_config\":{}}";

        testParseFromJsonString(expectedInputStrWithNullField, parsedInput -> {
            try {
                assertEquals(expectedOutputStr, serializationWithToXContent(parsedInput));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    @Test
    // Notice that this won't throw an IllegalStateException, which is pretty different from usual
    public void parseWithTestUserAndEmptyRateLimiterFieldWithNoException() throws Exception {
        final String expectedInputStrWithEmptyField = "{\"model_id\":\"testModelId\",\"user_rate_limiter_config\":" +
                "{\"testUser\":{}}}";
        final String expectedOutputStr = "{\"model_id\":\"testModelId\",\"user_rate_limiter_config\":" +
                "{}}";
        testParseFromJsonString(expectedInputStrWithEmptyField, parsedInput -> {
            try {
                assertEquals(expectedOutputStr, serializationWithToXContent(parsedInput));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    @Test
    public void parseWithNullField() throws Exception {
        exceptionRule.expect(IllegalStateException.class);
        final String expectedInputStrWithNullField = "{\"model_id\":null,\"user_rate_limiter_config\":" +
                "{\"testUser\":{\"rate_limit_number\":\"1\",\"rate_limit_unit\":\"MILLISECONDS\"}}}";

        testParseFromJsonString(expectedInputStrWithNullField, parsedInput -> {
            try {
                assertEquals(expectedInputStr, serializationWithToXContent(parsedInput));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    @Test
    public void parseWithIllegalField() throws Exception {
        final String expectedInputStrWithIllegalField = "{\"model_id\":\"testModelId\",\"illegal_field\":\"This field need to be skipped.\",\"user_rate_limiter_config\":" +
                "{\"testUser\":{\"rate_limit_number\":\"1\",\"rate_limit_unit\":\"MILLISECONDS\"}}}";

        testParseFromJsonString(expectedInputStrWithIllegalField, parsedInput -> {
            try {
                assertEquals(expectedInputStr, serializationWithToXContent(parsedInput));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    @Test
    // This will throw a ParsingException because MLRateLimiter parser cannot parse null field.
    public void parseWithNullMLRateLimiterInUserRateLimiterConfigFieldWithException() throws Exception {
        exceptionRule.expect(RuntimeException.class);
        final String expectedInputStrWithNullField = "{\"model_id\":\"testModelId\",\"user_rate_limiter_config\":{\"testUser\":null}}";
        final String expectedOutputStr = "{\"model_id\":\"testModelId\",\"user_rate_limiter_config\":{\"testUser\":null}}";

        testParseFromJsonString(expectedInputStrWithNullField, parsedInput -> {
            try {
                assertEquals(expectedOutputStr, serializationWithToXContent(parsedInput));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    @Test
    public void parseWithIllegalRateLimiterFieldWithException() throws Exception {
        exceptionRule.expect(RuntimeException.class);
        final String expectedInputStrWithIllegalField = "{\"model_id\":\"testModelId\",\"illegal_field\":\"This field need to be skipped.\",\"user_rate_limiter_config\":" +
                "{\"testUser\":\"Some illegal content that MLRateLimiter parser cannot parse.\"}}";

        testParseFromJsonString(expectedInputStrWithIllegalField, parsedInput -> {
            try {
                assertEquals(expectedInputStr, serializationWithToXContent(parsedInput));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    @Test
    public void testUserRateLimiterConfigUpdate() {
        MLRateLimiter rateLimiterWithNumber = MLRateLimiter.builder().limit(1.0).build();

        MLModelController modelControllerWithEmptyUserRateLimiterConfig = MLModelControllerGenerator();
        MLModelController modelControllerWithTestUserAndRateLimiterWithNumber = MLModelControllerGenerator("testUser", rateLimiterWithNumber);
        MLModelController modelControllerWithNewUserAndEmptyRateLimiter = MLModelControllerGenerator("newUser");

        modelControllerWithEmptyUserRateLimiterConfig.update(modelControllerNull);
        assertTrue(modelControllerWithEmptyUserRateLimiterConfig.getUserRateLimiterConfig().isEmpty());

        modelControllerWithEmptyUserRateLimiterConfig.update(modelControllerWithEmptyUserRateLimiterConfig);
        assertTrue(modelControllerWithEmptyUserRateLimiterConfig.getUserRateLimiterConfig().isEmpty());

        modelControllerWithEmptyUserRateLimiterConfig.update(modelControllerWithTestUserAndRateLimiterWithNumber);
        assertEquals("1", modelControllerWithEmptyUserRateLimiterConfig.getUserRateLimiterConfig().get("testUser").getLimit());
        assertNull(modelControllerWithEmptyUserRateLimiterConfig.getUserRateLimiterConfig().get("testUser").getUnit());

        modelControllerWithEmptyUserRateLimiterConfig.update(modelController);
        assertEquals("1", modelControllerWithEmptyUserRateLimiterConfig.getUserRateLimiterConfig().get("testUser").getLimit());
        assertEquals(TimeUnit.MILLISECONDS, modelControllerWithEmptyUserRateLimiterConfig.getUserRateLimiterConfig().get("testUser").getUnit());

        modelControllerWithEmptyUserRateLimiterConfig.update(modelControllerWithNewUserAndEmptyRateLimiter);
        assertTrue(modelControllerWithEmptyUserRateLimiterConfig.getUserRateLimiterConfig().get("newUser").isEmpty());
    }

    @Test
    public void testUserRateLimiterConfigIsUpdatable() {
        MLRateLimiter rateLimiterWithNumber = MLRateLimiter.builder().limit(1.0).build();

        MLModelController modelControllerWithEmptyUserRateLimiterConfig = MLModelControllerGenerator();
        MLModelController modelControllerWithTestUserAndRateLimiterWithNumber = MLModelControllerGenerator("testUser", rateLimiterWithNumber);
        MLModelController modelControllerWithNewUserAndRateLimiterWithNumber = MLModelControllerGenerator("newUser", rateLimiterWithNumber);
        MLModelController modelControllerWithNewUserAndEmptyRateLimiter = MLModelControllerGenerator("newUser");
        MLModelController modelControllerWithNewUserAndRateLimiter = MLModelControllerGenerator("newUser", rateLimiter);

        assertFalse(modelControllerWithEmptyUserRateLimiterConfig.isDeployRequiredAfterUpdate(null));
        assertFalse(modelControllerWithEmptyUserRateLimiterConfig.isDeployRequiredAfterUpdate(modelControllerNull));
        assertFalse(modelControllerWithEmptyUserRateLimiterConfig.isDeployRequiredAfterUpdate(modelControllerWithEmptyUserRateLimiterConfig));
        assertFalse(modelControllerWithEmptyUserRateLimiterConfig.isDeployRequiredAfterUpdate(modelControllerWithNewUserAndEmptyRateLimiter));

        assertFalse(modelControllerWithEmptyUserRateLimiterConfig.isDeployRequiredAfterUpdate(modelControllerWithTestUserAndRateLimiterWithNumber));
        assertFalse(modelControllerWithTestUserAndRateLimiterWithNumber.isDeployRequiredAfterUpdate(modelControllerWithTestUserAndRateLimiterWithNumber));
        assertTrue(modelControllerWithEmptyUserRateLimiterConfig.isDeployRequiredAfterUpdate(modelController));
        assertTrue(modelControllerWithTestUserAndRateLimiterWithNumber.isDeployRequiredAfterUpdate(modelController));

        assertFalse(modelControllerWithTestUserAndRateLimiterWithNumber.isDeployRequiredAfterUpdate(modelControllerWithNewUserAndRateLimiterWithNumber));
        assertTrue(modelControllerWithTestUserAndRateLimiterWithNumber.isDeployRequiredAfterUpdate(modelControllerWithNewUserAndRateLimiter));
    }

    private void testParseFromJsonString(String expectedInputStr, Consumer<MLModelController> verify) throws Exception {
        XContentParser parser = XContentType.JSON.xContent().createParser(new NamedXContentRegistry(new SearchModule(Settings.EMPTY,
                Collections.emptyList()).getNamedXContents()), LoggingDeprecationHandler.INSTANCE, expectedInputStr);
        parser.nextToken();
        MLModelController parsedInput = MLModelController.parse(parser);
        verify.accept(parsedInput);
    }

    private void readInputStream(MLModelController input, Consumer<MLModelController> verify) throws IOException {
        BytesStreamOutput bytesStreamOutput = new BytesStreamOutput();
        input.writeTo(bytesStreamOutput);
        StreamInput streamInput = bytesStreamOutput.bytes().streamInput();
        MLModelController parsedInput = new MLModelController(streamInput);
        verify.accept(parsedInput);
    }

    private String serializationWithToXContent(MLModelController input) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder();
        input.toXContent(builder, ToXContent.EMPTY_PARAMS);
        assertNotNull(builder);
        return builder.toString();
    }

    private MLModelController MLModelControllerGenerator(String user, MLRateLimiter rateLimiter) {
        return MLModelController.builder()
                .modelId("testModelId")
                .userRateLimiterConfig(new HashMap<>(){{put(user, rateLimiter);}})
                .build();

    }

    private MLModelController MLModelControllerGenerator(String user) {
        return MLModelController.builder()
                .modelId("testModelId")
                .userRateLimiterConfig(new HashMap<>(){{put(user, MLRateLimiter.builder().build());}})
                .build();

    }

    private MLModelController MLModelControllerGenerator() {
        return MLModelController.builder()
                .modelId("testModelId")
                .userRateLimiterConfig(new HashMap<>())
                .build();

    }

    @Ignore
    @Test
    public void testRateLimiterRemove() {
        MLModelController modelControllerWithTestUserAndEmptyRateLimiter = MLModelController.builder()
                .modelId("testModelId")
                .userRateLimiterConfig(new HashMap<>(){{put("testUser", MLRateLimiter.builder().build());}})
                .build();

        modelController.update(modelControllerWithTestUserAndEmptyRateLimiter);
        assertNull(modelController.getUserRateLimiterConfig().get("testUser"));
    }

}
