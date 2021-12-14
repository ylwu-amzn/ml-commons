/*
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  The OpenSearch Contributors require contributions made to
 *  this file be licensed under the Apache-2.0 license or a
 *  compatible open source license.
 *
 *  Modifications Copyright OpenSearch Contributors. See
 *  GitHub history for details.
 */

package org.opensearch.ml.common.output;

import org.opensearch.common.io.stream.Writeable;
import org.opensearch.common.xcontent.ToXContentObject;

public interface Output extends ToXContentObject, Writeable {

}
