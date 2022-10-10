/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.text_embedding;

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
import ai.djl.util.JsonUtils;
import com.google.gson.reflect.TypeToken;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.model.MLResultDataType;
import org.opensearch.ml.common.output.model.ModelTensor;

import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import  java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class SentenceTransformerTextEmbeddingTranslator implements ServingTranslator {
    public static final String TEXT_FIELDS = "text";
    private HuggingFaceTokenizer tokenizer;

    @Override
    public Batchifier getBatchifier() {
        return null;

    }
    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        Path path = ctx.getModel().getModelPath();
        tokenizer = HuggingFaceTokenizer.builder().optPadding(true).optTokenizerPath(path.resolve("tokenizer.json")).build();
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Input input) {
        Type listType = new TypeToken<ArrayList<String>>(){}.getType();
        List<String> sentences = JsonUtils.GSON.fromJson(input.getAsString(TEXT_FIELDS), listType);
        NDManager manager = ctx.getNDManager();
        NDList ndList = new NDList();
        Encoding[] encodings = tokenizer.batchEncode(sentences);
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
        return batchNDList;
    }

    @Override
    public Output processOutput(TranslatorContext ctx, NDList ndList) {
        Output output = new Output(200, "OK");
        List<ModelTensors> modelTensorOutputs = new ArrayList<>();
        NDList[] unbatchifiedNDList = Batchifier.STACK.unbatchify(ndList);
        for (NDList list : unbatchifiedNDList) {
            List<ModelTensor> outputs = new ArrayList<>();
            Iterator<NDArray> iterator = list.iterator();
            while (iterator.hasNext()) {
                NDArray ndArray = iterator.next();
                String name = ndArray.getName();
                Number[] data = ndArray.toArray();
                long[] shape = ndArray.getShape().getShape();
                DataType dataType = ndArray.getDataType();
                MLResultDataType mlResultDataType = MLResultDataType.valueOf(dataType.name());
                ByteBuffer buffer = ndArray.toByteBuffer();
                outputs.add(new ModelTensor(name, data, shape, mlResultDataType, buffer));
            }
            ModelTensors modelTensorOutput = new ModelTensors(outputs);
            modelTensorOutputs.add(modelTensorOutput);
        }
        try (BytesStreamOutput bytesStreamOutput = new BytesStreamOutput()) {
            bytesStreamOutput.writeInt(modelTensorOutputs.size());
            for (ModelTensors modelTensorOutput : modelTensorOutputs) {
                modelTensorOutput.writeTo(bytesStreamOutput);
            }
            bytesStreamOutput.flush();
            byte[] bytes = bytesStreamOutput.bytes().toBytesRef().bytes;
            output.add(bytes);
        } catch (Exception e) {
            throw new MLException("Failed to parse result");
        }
        return output;
    }

    @Override
    public void setArguments(Map<String, ?> arguments) {
    }
}
