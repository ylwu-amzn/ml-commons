/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.od;

import ai.djl.Application;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.tensorflow.engine.TfEngine;
import ai.djl.tensorflow.zoo.TfModelZoo;
import ai.djl.tensorflow.zoo.cv.objectdetction.TfSsdTranslator;
import ai.djl.training.util.ProgressBar;
import ai.djl.util.Pair;
import ai.djl.util.Platform;
import ai.djl.util.cuda.CudaUtils;
import io.protostuff.ProtostuffIOUtil;
import lombok.extern.log4j.Log4j2;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.Model;
import org.opensearch.ml.common.dataframe.DataFrame;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.input.parameter.MLAlgoParams;
import org.opensearch.ml.common.input.parameter.od.ObjectDetectionParams;
import org.opensearch.ml.common.input.parameter.sample.SampleAlgoParams;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.common.output.od.ObjectDetectionOutput;
import org.opensearch.ml.common.output.sample.SampleAlgoOutput;
import org.opensearch.ml.engine.Predictable;
import org.opensearch.ml.engine.Trainable;
import org.opensearch.ml.engine.annotation.Function;
import org.opensearch.ml.engine.utils.ModelSerDeSer;

import java.io.File;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

@Log4j2
@Function(FunctionName.OBJECT_DETECTION)
public class ObjectDetection implements Predictable {
    private static final int DEFAULT_SAMPLE_PARAM = -1;
    private String url;

    public ObjectDetection() {
    }

    public ObjectDetection(MLAlgoParams parameters) {
        this.url = ((ObjectDetectionParams) parameters).getUrl();
    }

    @Override
    public MLOutput predict(DataFrame dataFrame, Model model1) {
        AccessController.doPrivileged((PrivilegedAction<ObjectDetectionOutput>) () -> {
            ClassLoader contextClassLoader = Thread.currentThread().getContextClassLoader();
            try {
                System.setProperty("DJL_CACHE_DIR", "/home/ylwu/tmp");
                System.setProperty("java.library.path", "/home/ylwu/tf");
//                Thread.currentThread().setContextClassLoader(Application.class.getClassLoader());
//                Thread.currentThread().setContextClassLoader(Platform.class.getClassLoader());
                Thread.currentThread().setContextClassLoader(CudaUtils.class.getClassLoader());

                File file = new File("/usr/lib64", "libcudart.so");
                file = new File("/usr/lib64/libcudart.so");
                boolean exist = file.exists();
                Files.exists(Paths.get("/usr/lib64/libcudart.so"));
                Criteria<Image, DetectedObjects> criteria =
                        Criteria.builder()
                                .optApplication(Application.CV.OBJECT_DETECTION)
                                .setTypes(Image.class, DetectedObjects.class)
                                .optArtifactId("ssd") // download from S3
                                .optFilter("backbone", "mobilenet_v2")
//                                .optModelPath() // specify local file path
//                                .optModelUrls()// search from jar (pre-set context classl loader)/http,
                                .optEngine("TensorFlow")
                                .optProgress(new ProgressBar())
                                .build();

//        String url = "https://github.com/awslabs/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg";
                Image img = ImageFactory.getInstance().fromUrl(new URL(url));
                try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();
                     Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                    DetectedObjects result = predictor.predict(img);
                    List<String> classes =
                            result.items()
                                    .stream()
                                    .map(Classifications.Classification::getClassName)
                                    .collect(Collectors.toList());
                    return ObjectDetectionOutput.builder().objects(classes).build();
                }
            } catch (Exception e) {
                log.error("Failed to detect object from image", e);
                throw new MLException("Failed to detect object");
            } finally {
                Thread.currentThread().setContextClassLoader(contextClassLoader);
            }
        });
        return ObjectDetectionOutput.builder().objects(new ArrayList<>()).build();
    }

}
