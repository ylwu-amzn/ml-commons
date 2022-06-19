package org.opensearch.ml.engine.djl;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertToken;
import ai.djl.modality.nlp.bert.BertTokenizer;
import ai.djl.modality.nlp.qa.QAInput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.DownloadUtils;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.junit.Ignore;
import org.junit.Test;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class BertQaInferenceTest {

    /**
     * https://github.com/deepjavalibrary/djl/blob/master/jupyter/BERTQA.ipynb
     */
    @Test
    public void testQAMxNet() throws ModelNotFoundException, MalformedModelException, IOException, TranslateException {
//        System.setProperty("ai.djl.default_engine", "PyTorch");
        String question = "When did BBC Japan start broadcasting?";
        String resourceDocument = "BBC Japan was a general entertainment Channel.\n" +
                "Which operated between December 2004 and April 2006.\n" +
                "It ceased operations after its Japanese distributor folded.";

        QAInput input = new QAInput(question, resourceDocument);
        Criteria<QAInput, String> criteria = Criteria.builder()
                .optApplication(Application.NLP.QUESTION_ANSWER)
                .setTypes(QAInput.class, String.class)
                .optEngine("MXNet") // For DJL to use MXNet engine
                .optProgress(new ProgressBar()).build();
        ZooModel<QAInput, String> model = criteria.loadModel();

        Predictor<QAInput, String> predictor = model.newPredictor();
        String answer = predictor.predict(input);
        System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++");
        System.out.println(answer);
    }

    /**
     * https://github.com/deepjavalibrary/djl/blob/master/jupyter/pytorch/load_your_own_pytorch_bert.ipynb
     * @throws IOException
     * @throws ModelNotFoundException
     * @throws MalformedModelException
     * @throws TranslateException
     */
    @Test
    public void testLoadYourPyTorchModel() throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        System.setProperty("ai.djl.default_engine", "PyTorch");
        var question = "When did BBC Japan start broadcasting?";
        var resourceDocument = "BBC Japan was a general entertainment Channel.\n" +
                "Which operated between December 2004 and April 2006.\n" +
                "It ceased operations after its Japanese distributor folded.";


        URL url = getClass().getClassLoader().getResource("dog_bike_car.jpg");
        System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++ outputPath");
        String path = url.getPath();
        String outputPath = path.substring(0, path.lastIndexOf("/")) + "/bertqa/";
        System.out.println(outputPath);
        String modelPath = outputPath + "bertqa.pt";
        DownloadUtils.download("https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/pytorch/bertqa/0.0.1/trace_bertqa.pt.gz",
                outputPath + "bertqa.pt", new ProgressBar());
        BertTranslator translator = new BertTranslator();

        String vocabPath = outputPath + "vocab.txt";
        DownloadUtils.download("https://djl-ai.s3.amazonaws.com/mlrepo/model/nlp/question_answer/ai/djl/pytorch/bertqa/0.0.1/bert-base-uncased-vocab.txt.gz",
                vocabPath, new ProgressBar());


        Criteria<QAInput, String> criteria = Criteria.builder()
                .setTypes(QAInput.class, String.class)
                .optModelPath(Paths.get(outputPath)) // search in local folder
                .optTranslator(translator)
                .optEngine("PyTorch")
                .optProgress(new ProgressBar()).build();

        ZooModel model = criteria.loadModel();
        String predictResult = null;
        QAInput input = new QAInput(question, resourceDocument);

// Create a Predictor and use it to predict the output
        try (Predictor<QAInput, String> predictor = model.newPredictor(translator)) {
            predictResult = predictor.predict(input);
        }

        System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++ output");
        System.out.println(question);
        System.out.println(predictResult);
    }

    class BertTranslator implements Translator<QAInput, String> {
        private List<String> tokens;
        private Vocabulary vocabulary;
        private BertTokenizer tokenizer;

        @Override
        public void prepare(TranslatorContext ctx) throws IOException {
            Path path = Paths.get("/Users/ylwu/code/os/ml-commons/ml-algorithms/build/resources/test/bertqa/vocab.txt");
            vocabulary = DefaultVocabulary.builder()
                    .optMinFrequency(1)
                    .addFromTextFile(path)
                    .optUnknownToken("[UNK]")
                    .build();
            tokenizer = new BertTokenizer();
        }

        @Override
        public NDList processInput(TranslatorContext ctx, QAInput input) {
            BertToken token =
                    tokenizer.encode(
                            input.getQuestion().toLowerCase(),
                            input.getParagraph().toLowerCase());
            // get the encoded tokens that would be used in precessOutput
            tokens = token.getTokens();
            NDManager manager = ctx.getNDManager();
            // map the tokens(String) to indices(long)
            long[] indices = tokens.stream().mapToLong(vocabulary::getIndex).toArray();
            long[] attentionMask = token.getAttentionMask().stream().mapToLong(i -> i).toArray();
            long[] tokenType = token.getTokenTypes().stream().mapToLong(i -> i).toArray();
            NDArray indicesArray = manager.create(indices);
            NDArray attentionMaskArray =
                    manager.create(attentionMask);
            NDArray tokenTypeArray = manager.create(tokenType);
            // The order matters
            return new NDList(indicesArray, attentionMaskArray, tokenTypeArray);
        }

        @Override
        public String processOutput(TranslatorContext ctx, NDList list) {
            NDArray startLogits = list.get(0);
            NDArray endLogits = list.get(1);
            int startIdx = (int) startLogits.argMax().getLong();
            int endIdx = (int) endLogits.argMax().getLong();
            return tokens.subList(startIdx, endIdx + 1).toString();
        }

        @Override
        public Batchifier getBatchifier() {
            return Batchifier.STACK;
        }
    }

    /**
     * https://github.com/deepjavalibrary/djl/blob/master/jupyter/load_pytorch_model.ipynb
     */
    @Test
    public void testLoadPyTorchModel() throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {
        System.setProperty("ai.djl.default_engine", "PyTorch");
        URL url = getClass().getClassLoader().getResource("dog_bike_car.jpg");
        System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++");
        String path = url.getPath();
        String outputPath = path.substring(0, path.lastIndexOf("/"))+ "/pytorch_models/resnet18/";
        System.out.println(outputPath);
        String modelPath = outputPath + "resnet18.pt";
        DownloadUtils.download("https://djl-ai.s3.amazonaws.com/mlrepo/model/cv/image_classification/ai/djl/pytorch/resnet/0.0.1/traced_resnet18.pt.gz",
                modelPath, new ProgressBar());

        String synsetPath = outputPath + "synset.txt";
        DownloadUtils.download("https://djl-ai.s3.amazonaws.com/mlrepo/model/cv/image_classification/ai/djl/pytorch/synset.txt",
                synsetPath, new ProgressBar());

        Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
                .addTransform(new Resize(256))
                .addTransform(new CenterCrop(224, 224))
                .addTransform(new ToTensor())
                .addTransform(new Normalize(
                        new float[] {0.485f, 0.456f, 0.406f},
                        new float[] {0.229f, 0.224f, 0.225f}))
                .optApplySoftmax(true)
                .build();

        Criteria<Image, Classifications> criteria = Criteria.builder()
                .setTypes(Image.class, Classifications.class)
                .optModelPath(Paths.get(outputPath))
                .optOption("mapLocation", "true") // this model requires mapLocation for GPU
                .optTranslator(translator)
                .optEngine("PyTorch")
                .optProgress(new ProgressBar()).build();

        ZooModel model = criteria.loadModel();
        var img = ImageFactory.getInstance().fromUrl("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg");
        Predictor<Image, Classifications> predictor = model.newPredictor();
        Classifications classifications = predictor.predict(img);
        System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++ output");
        System.out.println(classifications);
    }
}
