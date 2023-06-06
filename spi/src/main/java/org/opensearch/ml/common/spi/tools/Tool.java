/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.spi.tools;

public interface Tool {

    <T> T run(String input);

    String getName();
    String getDescription();

    boolean validate(String input);
    default boolean end(String input){return false;}

}
