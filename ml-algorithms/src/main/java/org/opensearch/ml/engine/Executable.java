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

package org.opensearch.ml.engine;

import org.opensearch.ml.common.input.Input;
import org.opensearch.ml.common.output.Output;

public interface Executable {

    /**
     * Execute algorithm with given input data.
     * @param input input data
     * @return execution result
     */
    Output execute(Input input);

}
