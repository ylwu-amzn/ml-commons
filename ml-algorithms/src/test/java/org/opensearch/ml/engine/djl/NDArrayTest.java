package org.opensearch.ml.engine.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.junit.Test;

public class NDArrayTest {

    @Test
    public void testA() {
        // https://zhuanlan.zhihu.com/c_1255493231133417472
        // https://zhuanlan.zhihu.com/p/213785439
        try(NDManager manager = NDManager.newBaseManager()) {
            NDArray nd = manager.ones(new Shape(2, 3));
            System.out.println(nd);

            NDArray nd2 = manager.randomUniform(0, 1, new Shape(1, 1, 4));
            System.out.println(nd2);

            System.out.println("++++++++++++++++++++++++++++++ transpose");
            NDArray nd3 = manager.arange(1, 10).reshape(3, 3);
            System.out.println(nd3);
            nd3 = nd3.transpose();
            System.out.println(nd3);
            nd3 = nd3.add(10);
            System.out.println(nd3);
        }
    }
}
