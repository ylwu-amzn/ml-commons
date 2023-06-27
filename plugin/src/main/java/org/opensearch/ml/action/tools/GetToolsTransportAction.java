package org.opensearch.ml.action.tools;

import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;
import lombok.extern.log4j.Log4j2;
import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.ml.common.ToolMetadata;
import org.opensearch.ml.common.transport.tools.MLGetToolsAction;
import org.opensearch.ml.common.transport.tools.MLToolsGetRequest;
import org.opensearch.ml.common.transport.tools.MLToolsGetResponse;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Log4j2
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class GetToolsTransportAction extends HandledTransportAction<ActionRequest, MLToolsGetResponse> {
    @Inject
    public GetToolsTransportAction(
            TransportService transportService,
            ActionFilters actionFilters
    ) {
        super(MLGetToolsAction.NAME, transportService, actionFilters, MLToolsGetRequest::new);
    }
    /**
     * @param task
     * @param request
     * @param listener
     */
    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<MLToolsGetResponse> listener) {
        MLToolsGetRequest mlToolsGetRequest = MLToolsGetRequest.fromActionRequest(request);

        List<ToolMetadata> externalTools = mlToolsGetRequest.getExternalTools();
        List<ToolMetadata> toolsList = new ArrayList<>(
                Arrays.asList(
                        ToolMetadata.builder()
                                .name("LanguageModelTool")
                                .description("Useful for answering any general questions.")
                                .build(),
                        ToolMetadata.builder()
                                .name("MathTool")
                                .description("Use this tool to calculate any math problem.")
                                .build(),
                        ToolMetadata.builder()
                                .name("SearchIndexTool")
                                .description("Useful for when you don't know answer for some question or need to search my private data in OpenSearch index.")
                                .build(),
                        ToolMetadata.builder()
                                .name("SearchWikipediaTool")
                                .description("Useful when you need to use this tool to search general knowledge on wikipedia.")
                                .build()
                )
        );
        toolsList.addAll(externalTools);
        try {
            listener.onResponse(MLToolsGetResponse.builder()
                    .toolMetadata(toolsList)
                    .build());
        } catch (Exception e) {
            log.error("Failed to get tools list", e);
            listener.onFailure(e);
        }
    }
}
