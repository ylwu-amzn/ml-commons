/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.breaker;

import org.opensearch.monitor.jvm.JvmService;

/**
 * A circuit breaker for memory usage.
 */
public class MemoryCircuitBreaker extends ThresholdCircuitBreaker<Short> {

    public static final short DEFAULT_JVM_HEAP_USAGE_THRESHOLD = 85;
    private final JvmService jvmService;

    public MemoryCircuitBreaker(JvmService jvmService) {
        super(DEFAULT_JVM_HEAP_USAGE_THRESHOLD);
        this.jvmService = jvmService;
    }

    public MemoryCircuitBreaker(short threshold, JvmService jvmService) {
        super(threshold);
        this.jvmService = jvmService;
    }

    @Override
    public boolean isOpen() {
        return jvmService.stats().getMem().getHeapUsedPercent() > this.getThreshold();
    }
}
