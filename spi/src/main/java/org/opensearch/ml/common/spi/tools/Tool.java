/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.spi.tools;

import java.util.Map;

public interface Tool {

    <T> T run(String input, Map<String, String> toolParameters);

    String getName();
    String getDescription();

    boolean validate(String input, Map<String, String> toolParameters);
    default boolean end(String input, Map<String, String> toolParameters){return false;}

}
