package org.opensearch.ml.test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class TestMultiThread {

    public static AtomicInteger atomInt = new AtomicInteger(0);


    public static void main(String[] args) throws InterruptedException {
        List<ThreadDemo> threads = new ArrayList<>();
        for (int i=0;i<1000; i++) {
            threads.add(new ThreadDemo("t"+i, atomInt));
        }

//        threads.parallelStream().forEach(e -> e.start());
        threads.stream().parallel().forEach(e -> e.start());
        Thread.sleep(5000);
    }


}
