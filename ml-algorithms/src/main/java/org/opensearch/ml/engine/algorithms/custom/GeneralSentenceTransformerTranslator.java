/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.custom;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Batchifier;
import ai.djl.translate.ServingTranslator;
import ai.djl.translate.TranslatorContext;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.model.MLResultDataType;
import org.opensearch.ml.common.output.custom_model.MLModelTensor;
import org.opensearch.ml.common.output.custom_model.MLModelTensorOutput;

import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class GeneralSentenceTransformerTranslator implements ServingTranslator {
//    public static final int SIZE_LIMIT = 512;
    private HuggingFaceTokenizer tokenizer;

    @Override
    public Batchifier getBatchifier() {
        return null;

    }
    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        Path path = ctx.getModel().getModelPath();
//        [a]
//        [a,b]
//        [a, b,c ]
//Some huggingface model's tokenizer.json is not good. Some tokenizer logic is in their python code, not in tokenizer like no truncation      https://huggingface.co/sentence-transformers/msmarco-roberta-base-ance-firstp
        tokenizer = HuggingFaceTokenizer.builder().optPadding(true).optTokenizerPath(path.resolve("tokenizer.json")).build();
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Input input) {
        Gson gson = new Gson();
//        List<String> sentences = Arrays.asList(input.getAsString("docs"));
        Type listType = new TypeToken<ArrayList<String>>(){}.getType();
        List<String> sentences = gson.fromJson(input.getAsString("docs"), listType);
//        String sentence = input.getAsString("doc");
        NDManager manager = ctx.getNDManager();
        NDList ndList = new NDList();
        //TODO: support batch request
//        Encoding[] encodings = tokenizer.batchEncode(sentences);
//        encodings[0].getIds();
//        Shape shape = new Shape(sentences.size(), 512);
//        NDArray ndArray = manager.create(shape);
//        sentence = sentence.strip().toLowerCase(Locale.ROOT);

//        List<String> sentences = Arrays.asList(sentence);

        Encoding[] encodings = tokenizer.batchEncode(sentences);
//        Encoding encode = tokenizer.encode(sentence);

        NDList[] ndLists = new NDList[sentences.size()];
        for (int i=0; i<encodings.length; i++) {
            Encoding encode = encodings[i];
            long[] ids = encode.getIds();
            long[] attentionMask = encode.getAttentionMask();

            NDArray indicesArray;
            indicesArray = manager.create(ids);
            indicesArray.setName("input1.input_ids");

            NDArray attentionMaskArray;
            attentionMaskArray = manager.create(attentionMask);
            attentionMaskArray.setName("input1.attention_mask");
            ndList.add(indicesArray);
            ndList.add(attentionMaskArray);

            ndLists[i] = (ndList);
        }
        NDList batchNDList = Batchifier.STACK.batchify(ndLists);


//        long[] indices = encode.getIds();
//        NDArray indicesArray;
//        indicesArray = manager.create(indices);
//        indicesArray.setName("input1.input_ids");// [ [11,1,1], [1] ]
//
//        long[] attentionMask = encode.getAttentionMask();
//        NDArray attentionMaskArray;
//        attentionMaskArray = manager.create(attentionMask);
//        attentionMaskArray.setName("input1.attention_mask");
//        ndList.add(indicesArray);
//        ndList.add(attentionMaskArray);
        return batchNDList;
    }

    @Override
    public Output processOutput(TranslatorContext ctx, NDList ndList) {
        /**
         * Binary: squeeze [x, x, x, x,x ,x ,x]
         * Binary: array.shape [1, 2, 2]
         * NDarray.reshape()
         */
//        List<float[]> embeddings = new ArrayList<>();

        Output output = new Output(200, "OK");
        List<MLModelTensorOutput> modelTensorOutputs = new ArrayList<>();
        NDList[] unbatchifiedNDList = Batchifier.STACK.unbatchify(ndList);
        for (NDList list : unbatchifiedNDList) {
            List<MLModelTensor> outputs = new ArrayList<>();
            Iterator<NDArray> iterator = list.iterator();
            while (iterator.hasNext()) {
                NDArray ndArray = iterator.next();
                String name = ndArray.getName();//could be null
                DataType dataType = ndArray.getDataType();// not null ----------------------1
                String device = ndArray.getDevice().toString();// no null
                long[] shape = ndArray.getShape().getShape();// not null, could be 0 -------2
                // callout limit: only float /int
                // add one option to return bytes directly.
//            byte[] bytes = ndArray.getAsBytes(); //-------------------------3.2
//            if (dataType == DataType.FLOAT32 && name.equals("sentence_embedding")) {
//                float[] floats = byteToFloat(bytes);
//                //System.out.println(floats);
//
//                ByteBuffer buffer = ByteBuffer.wrap(bytes);
//                FloatBuffer fb = buffer.asFloatBuffer();
//                float[] ret = new float[fb.remaining()];
//                fb.get(ret);
//                System.out.println(ret);
//            }
                ByteBuffer buffer = ndArray.toByteBuffer();
                if (dataType == DataType.FLOAT32 && name.equals("sentence_embedding")) {
                    FloatBuffer fb = buffer.asFloatBuffer();
                    float[] ret = new float[fb.remaining()];
                    fb.get(ret);

                    byte[] array = new byte[buffer.remaining()];
                    buffer.get(array);

                    ByteBuffer buffer2 = ByteBuffer.wrap(array);
                    buffer2.order(buffer.order());
                    FloatBuffer fb2 = buffer2.asFloatBuffer();
                    float[] ret2 = new float[fb2.remaining()];
                    fb2.get(ret2);

                    ByteBuffer buffer3 = ByteBuffer.wrap(array);
                    FloatBuffer fb3 = buffer3.asFloatBuffer();
                    float[] ret3 = new float[fb3.remaining()];
                    fb3.get(ret3);
                    boolean a = ret3 == ret2;
                }
                Number[] data = ndArray.toArray();// string can't use to Array -------------3.1
//             ByteBuffer byteBuffer = ndArray.toByteBuffer(); //-------------------------3.2
//            ndArray.toByteBuffer();
                // bytes of ndArray storing same data type and continuously
                // byte[] asBytes = ndArray.getAsBytes();// base64 for REST API?

//            MLResultDataType mlResultDataType = MLResultDataType.UNKNOWN;
//            if (dataType.isFloating()) {
//                mlResultDataType = MLResultDataType.FLOAT;
//            } else if (dataType.isInteger()) {
//                mlResultDataType = MLResultDataType.INT;
//            } else if (dataType.isBoolean()) {
//                mlResultDataType = MLResultDataType.BOOLEAN;
//            }
                MLResultDataType mlResultDataType = MLResultDataType.valueOf(dataType.name());

                outputs.add(new MLModelTensor(name, data, shape, mlResultDataType, device, buffer));
            }
            MLModelTensorOutput modelTensorOutput = new MLModelTensorOutput(outputs);
            modelTensorOutputs.add(modelTensorOutput);
//            try (BytesStreamOutput bytesStreamOutput = new BytesStreamOutput()) {
//                modelTensorOutput.writeTo(bytesStreamOutput);
//                bytesStreamOutput.flush();
//                byte[] bytes = bytesStreamOutput.bytes().toBytesRef().bytes;
//                output.add(bytes);
//            } catch (Exception e) {
//                throw new MLException("Failed to parse result");
//            }
        }
        try (BytesStreamOutput bytesStreamOutput = new BytesStreamOutput()) {
            bytesStreamOutput.writeInt(modelTensorOutputs.size());
            for (MLModelTensorOutput modelTensorOutput : modelTensorOutputs) {
                modelTensorOutput.writeTo(bytesStreamOutput);
            }
            bytesStreamOutput.flush();
            byte[] bytes = bytesStreamOutput.bytes().toBytesRef().bytes;
            output.add(bytes);
        } catch (Exception e) {
            throw new MLException("Failed to parse result");
        }


//        output.add(BytesUtils.floatArrayToBytes(embedding));
        return output;
    }

    public static float[] byteToFloat(byte[] input) {
        float[] ret = new float[input.length/4];
        for (int x = 0; x < input.length; x+=4) {
            ret[x/4] = ByteBuffer.wrap(input, x, 4).getFloat();
        }
        return ret;
    }

    public static byte[] floatToByte(float[] input) {
        byte[] ret = new byte[input.length*4];
        for (int x = 0; x < input.length; x++) {
            ByteBuffer.wrap(ret, x*4, 4).putFloat(input[x]);
        }
        return ret;
    }

//    @Override
//    public Batchifier getBatchifier() {
//        return Batchifier.STACK;
//    }

    @Override
    public void setArguments(Map<String, ?> arguments) {
    }
}
