package org.opensearch.ml.test;

import java.util.concurrent.atomic.AtomicInteger;

class ThreadDemo extends Thread {
    private Thread t;
    private String threadName;
    private AtomicInteger autoInt;

    ThreadDemo(String name, AtomicInteger autoInt) {
        threadName = name;
        this.autoInt = autoInt;
        System.out.println("Creating " +  threadName );
    }

    public void run() {
//        int val = autoInt.getAndIncrement();

        /*int val = autoInt.getAndIncrement();
        int newVal = val;
        if (val > 4 -1) {
            newVal = 0;
            autoInt.set(newVal + 1);
        }*/

        int val = autoInt.getAndIncrement();
        int newVal = val;
        if (val > 4 -1) {
            newVal = val % 4;
            autoInt.set(newVal + 1);
        }
        System.out.println(threadName + " --- " + newVal);
    }

    public void start () {
        if (t == null) {
            t = new Thread (this, threadName);
            t.start ();
        }
    }
}
