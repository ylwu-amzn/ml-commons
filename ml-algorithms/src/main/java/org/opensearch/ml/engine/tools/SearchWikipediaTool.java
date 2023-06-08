package org.opensearch.ml.engine.tools;

import com.google.gson.Gson;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.util.EntityUtils;
import org.opensearch.action.ActionFuture;
import org.opensearch.action.ActionRequest;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.dataset.MLInputDataset;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.spi.tools.ToolAnnotation;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;

import java.security.AccessController;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.opensearch.ml.engine.tools.ToolSettings.SUMMARY_MODEL_ID;


//@ToolAnnotation(SearchWikipediaTool.NAME)
public class SearchWikipediaTool implements Tool {
    public static final String NAME = "SearchWikipediaTool";

    private static SearchWikipediaTool INSTANCE;

    public static SearchWikipediaTool getInstance() {
        if (INSTANCE != null) {
            return INSTANCE;
        }
        synchronized (SearchWikipediaTool.class) {
            if (INSTANCE != null) {
                return INSTANCE;
            }
            INSTANCE = new SearchWikipediaTool();
            return INSTANCE;
        }
    }

    private volatile String summaryModelId;

    private SearchWikipediaTool() {

    }

    public void init() {
        summaryModelId = SUMMARY_MODEL_ID.get(settings);
        clusterService
                .getClusterSettings()
                .addSettingsUpdateConsumer(SUMMARY_MODEL_ID, it -> summaryModelId = it);
    }

    private Client client;
    private Settings settings;
    private ClusterService clusterService;
    private static final String description = "Use this tool to search general knowledge on wikipedia.";

    @Override
    public <T> T run(String input) {
        HttpClient httpClient = HttpClientBuilder.create().build();
        //String title = "English";
        //String title = "Pet_Door";
        //String title = "Atmosphere_of_Earth";
        String title = input.trim().replace(" ", "_");
        title = title.replace("\"", "");
        HttpGet httpGet = new HttpGet("https://en.wikipedia.org/w/api.php?action=query&prop=revisions&titles=" + title + "&rvslots=*&rvprop=content&formatversion=2&format=json");
        try {
            String pageContentSummary = AccessController.doPrivileged((PrivilegedExceptionAction<String>) () -> {
                HttpResponse httpResponse = httpClient.execute(httpGet);
                String content = EntityUtils.toString(httpResponse.getEntity());
                Gson gson = new Gson();
                Map map = gson.fromJson(content, Map.class);
                if (map.containsKey("query")) {
                    Map<String, Object> queryMap = (Map<String, Object>)map.get("query");
                    List pages = (List) queryMap.get("pages");
                    content = gson.toJson(pages.get(0));
                }
                //content = content.replaceAll("[^\\p{L}\\p{N}]+", " ");
                int limit = 700;
                int overlap = 100;
                // Split the string into words
                String[] words = content.split("\\s+");

                int processedWords = 0;

                //String modelId = "VMSPCYgBHGC7TZ5evvnF";
                List<String> summaries = new ArrayList<>();
                int maxSummaries = 0;
                while (processedWords < words.length && maxSummaries < 5) {
                    // Join the desired number of words together
                    int to = Math.min(processedWords + limit + overlap, words.length);
                    String[] tmpWords = Arrays.copyOfRange(words, processedWords, to);
                    String result = String.join(" ", tmpWords);

                    Map<String, String> parameters = new HashMap<>();
                    parameters.put("prompt", "Summarize the input document. The output must be less than 100 words long. \n\n Input: ${parameters.input} ");
                    parameters.put("input", gson.toJson(result));
                    MLInputDataset inputDataSet = RemoteInferenceInputDataSet.builder().parameters(parameters).build();
                    MLInput mlInput = MLInput.builder().algorithm(FunctionName.REMOTE).inputDataset(inputDataSet).build();
                    MLPredictionTaskRequest request = new MLPredictionTaskRequest(summaryModelId, mlInput);
                    ActionFuture<MLTaskResponse> taskResponse = client.execute(MLPredictionTaskAction.INSTANCE, request);
                    MLTaskResponse mlTaskResponse = taskResponse.actionGet(30, TimeUnit.SECONDS);
                    //ActionFuture<MLOutput> output = mlClient.predict(summaryModelId, mlInput);
                    ModelTensorOutput mlOutput = (ModelTensorOutput)mlTaskResponse.getOutput();
                    //ModelTensorOutput mlOutput = (ModelTensorOutput)output.actionGet(30, TimeUnit.SECONDS);
                    String summary = (String)mlOutput.getMlModelOutputs().get(0).getMlModelTensors().get(0).getDataAsMap().get("response");
                    summaries.add(summary);
                    processedWords += limit;
                    maxSummaries ++;
                }

                // Join the desired number of words together
                Map<String, String> parameters = new HashMap<>();
                parameters.put("prompt", "Summarize the input document. The output must be less than 1000 characters long. \n\n Input: ${parameters.input} ");
                parameters.put("input", "\"" + gson.toJson(summaries) + "\"");
                MLInputDataset inputDataSet = RemoteInferenceInputDataSet.builder().parameters(parameters).build();
                MLInput mlInput = MLInput.builder().algorithm(FunctionName.REMOTE).inputDataset(inputDataSet).build();

                MLPredictionTaskRequest request = new MLPredictionTaskRequest(summaryModelId, mlInput);
                ActionFuture<MLTaskResponse> taskResponse = client.execute(MLPredictionTaskAction.INSTANCE, request);
                MLTaskResponse mlTaskResponse = taskResponse.actionGet(30, TimeUnit.SECONDS);
                ModelTensorOutput mlOutput = (ModelTensorOutput)mlTaskResponse.getOutput();
//                ActionFuture<MLOutput> output = mlClient.predict(summaryModelId, mlInput);
//                ModelTensorOutput mlOutput = (ModelTensorOutput)output.actionGet(30, TimeUnit.SECONDS);
                String summary = (String)mlOutput.getMlModelOutputs().get(0).getMlModelTensors().get(0).getDataAsMap().get("response");

                return summary;
            });
            return (T)pageContentSummary;
        } catch (Exception e) {
            return null;
        }
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public String getDescription() {
        return description;
    }

    @Override
    public boolean validate(String s) {
        return s != null && s.length() > 0;
    }

    public void setClient(Client client) {
        this.client = client;
    }

    public void setSettings(Settings settings) {
        this.settings = settings;
    }

    public void setClusterService(ClusterService clusterService) {
        this.clusterService = clusterService;
    }
}
