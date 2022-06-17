package org.opensearch.ml.engine.djl;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class ObjectDetectionTest {

    /**
     * https://www.zhihu.com/question/412453568/answer/1390351776
     * DJL code TfSsdTest.java
     */
    @Test
    public void testTfSSD() throws IOException, ModelException, TranslateException {

        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optArtifactId("ssd")
                        .optFilter("backbone", "mobilenet_v2")
                        .optEngine("TensorFlow")
                        .optProgress(new ProgressBar())
                        .build();

        String url = "https://github.com/awslabs/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg";
//        Path file = Paths.get("/Users/ylwu/code/os/djl/examples/src/test/resources/dog_bike_car.jpg");
//        Path file = Paths.get("/Users/ylwu/Desktop/dog_frog.png");

//        ClassLoader classLoader = getClass().getClassLoader();
//        String path = classLoader.getResource("dog_bike_car.jpg").getPath();
//        Path file = Paths.get(path);
//        Image img = ImageFactory.getInstance().fromFile(file);
        Image img = ImageFactory.getInstance().fromUrl(new URL(url));
        try (ZooModel<Image, DetectedObjects> model = criteria.loadModel();
             Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
            for (Pair<String, Shape> pair : model.describeOutput()) {
                if (pair.getKey().contains("label")) {
                    Assert.assertEquals(pair.getValue(), new Shape(-1, 1));
                } else if (pair.getKey().contains("box")) {
                    Assert.assertEquals(pair.getValue(), new Shape(-1, 4));
                } else if (pair.getKey().contains("score")) {
                    Assert.assertEquals(pair.getValue(), new Shape(-1, 1));
                } else {
                    throw new IllegalStateException("Unexpected output name:" + pair.getKey());
                }
            }

            DetectedObjects result = predictor.predict(img);
            List<String> classes =
                    result.items()
                            .stream()
                            .map(Classifications.Classification::getClassName)
                            .collect(Collectors.toList());
//            Assert.assertTrue(classes.contains("Dog"));
//            Assert.assertTrue(classes.contains("Bicycle"));
//            Assert.assertTrue(classes.contains("Car"));
            System.out.printf("++++++++++++++++++++");
            System.out.println(Arrays.toString(classes.toArray(new String[0])));
//            saveBoundingBoxImage(img, result);
        }

//        private static void saveBoundingBoxImage(Image img, DetectedObjects detection) throws IOException {
//            Path outputDir = Paths.get("build/output");
//            Files.createDirectories(outputDir);
//
//            img.drawBoundingBoxes(detection);
//
//            Path imagePath = outputDir.resolve("detected-dog_bike_car.png");
//            // OpenJDK can't save jpg with alpha channel
//            img.save(Files.newOutputStream(imagePath), "png");
//        }
    }
}
