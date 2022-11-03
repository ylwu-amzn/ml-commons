/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.breaker;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.monitor.jvm.JvmService;
import org.opensearch.monitor.jvm.JvmStats;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicInteger;

import static org.mockito.Mockito.when;

public class MemoryCircuitBreakerTests {

    @Mock
    JvmService jvmService;

    @Mock
    JvmStats jvmStats;

    @Mock
    JvmStats.Mem mem;

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
        when(jvmService.stats()).thenReturn(jvmStats);
        when(jvmStats.getMem()).thenReturn(mem);
        when(mem.getHeapUsedPercent()).thenReturn((short) 50);
    }

    @Test
    public void testIsOpen() {
        // default threshold 85%
        CircuitBreaker breaker = new MemoryCircuitBreaker(jvmService);
        Assert.assertFalse(breaker.isOpen());

        // custom threshold 90%
        breaker = new MemoryCircuitBreaker((short) 90, jvmService);
        Assert.assertFalse(breaker.isOpen());
    }

    @Test
    public void testIsOpen_ExceedMemoryThreshold() {
        CircuitBreaker breaker = new MemoryCircuitBreaker(jvmService);

        when(mem.getHeapUsedPercent()).thenReturn((short) 95);
        Assert.assertTrue(breaker.isOpen());
    }

    @Test
    public void testIsOpen_CustomThreshold_ExceedMemoryThreshold() {
        CircuitBreaker breaker = new MemoryCircuitBreaker((short) 90, jvmService);

        when(mem.getHeapUsedPercent()).thenReturn((short) 95);
        Assert.assertTrue(breaker.isOpen());
    }

    @Test
    public void testA() {
        ConcurrentLinkedDeque<Integer> queue = new ConcurrentLinkedDeque();
        queue.add(0);
        queue.add(1);
        queue.add(2);
        queue.add(3);
        long s = System.nanoTime();
        for(int i=0;i<1000_000;i++){
            Integer item = queue.poll();
            queue.add(item);
        }
        long e = System.nanoTime(); // 195 105  108 98.873676
        System.out.println((e -s)/1e6);
    }

    @Test
    public void testB() {
        AtomicInteger nextDevice = new AtomicInteger(0);
        nextDevice.compareAndSet(0, 2);
//        System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++");
//        System.out.println(nextDevice.get());
//        System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++");
        long s = System.nanoTime();
        for(int i=0;i<1000_000;i++){
            int currentDevice = nextDevice.incrementAndGet();
            if (currentDevice > 4 - 1) {
                int actualCurrentDevice = currentDevice % 4;
                //nextDevice.set(currentDevice + 1);
                nextDevice.compareAndSet(currentDevice, (actualCurrentDevice + 1) % 4);
            }
        }
        long e = System.nanoTime(); // 16.10529 20.60999 24.238489 29.86013
        System.out.println((e -s)/1e6);
    }
}
