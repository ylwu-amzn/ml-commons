package org.opensearch.ml.engine.utils;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Roudrobin load balancer.
 * TODO:// refactor to support more load balancer.
 */
public class RoundRobinLoadBalancer {

    private final int maxNodes;
    private final AtomicInteger currentNode;

    public RoundRobinLoadBalancer(int maxNodes) {
        this.maxNodes = maxNodes;
        this.currentNode = new AtomicInteger(-1);
    }

    public int getNext() {
        int next = currentNode.incrementAndGet();
        if (next > maxNodes - 1) {
            int recalculatedNext = next % maxNodes;
            currentNode.compareAndSet(next, recalculatedNext);
            return recalculatedNext;
        }
        return next;
    }

    public static int getNext() {
        roundRobinLoadBalancer
    }
}
