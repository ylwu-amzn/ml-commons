/*
 * SPDX-License-Identifier: Apache-2.0
/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.stats.suppliers;

import org.opensearch.ml.utils.IndexUtils;

import java.util.function.Supplier;

/**
 * IndexStatusSupplier provides the status of an index as the value
 */
public class IndexStatusSupplier implements Supplier<String> {
    private IndexUtils indexUtils;
    private String indexName;

    public static final String UNABLE_TO_RETRIEVE_HEALTH_MESSAGE = "unable to retrieve health";

    /**
     * Constructor
     *
     * @param indexUtils Utility for getting information about indices
     * @param indexName Name of index to extract stats from
     */
    public IndexStatusSupplier(IndexUtils indexUtils, String indexName) {
        this.indexUtils = indexUtils;
        this.indexName = indexName;
    }

    @Override
    public String get() {
        try {
            return indexUtils.getIndexHealthStatus(indexName);
        } catch (IllegalArgumentException e) {
            return UNABLE_TO_RETRIEVE_HEALTH_MESSAGE;
        }

    }
}
