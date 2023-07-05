/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.models;

import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;
import lombok.extern.log4j.Log4j2;
import org.opensearch.action.ActionListener;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.get.GetRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.action.update.UpdateRequest;
import org.opensearch.client.Client;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.IndexNotFoundException;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.connector.Connector;
import org.opensearch.ml.common.connector.HttpConnector;
import org.opensearch.ml.common.exception.MLResourceNotFoundException;
import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.common.transport.model.MLModelUpdateAction;
import org.opensearch.ml.common.transport.model.MLModelUpdateRequest;
import org.opensearch.ml.common.transport.model.MLModelUpdateResponse;
import org.opensearch.ml.common.transport.register.MLUpdateModelInput;
import org.opensearch.ml.engine.MLEngine;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.common.CommonValue.ML_MODEL_INDEX;
import static org.opensearch.ml.utils.MLNodeUtils.createXContentParserFromRegistry;

@Log4j2
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class UpdateModelTransportAction extends HandledTransportAction<ActionRequest, MLModelUpdateResponse> {

    Client client;
    NamedXContentRegistry xContentRegistry;
    MLEngine mlEngine;

    @Inject
    public UpdateModelTransportAction(
        TransportService transportService,
        ActionFilters actionFilters,
        Client client,
        NamedXContentRegistry xContentRegistry,
        MLEngine mlEngine
    ) {
        super(MLModelUpdateAction.NAME, transportService, actionFilters, MLModelUpdateRequest::new);
        this.client = client;
        this.xContentRegistry = xContentRegistry;
        this.mlEngine = mlEngine;
    }

    @Override
    protected void doExecute(Task task, ActionRequest request, ActionListener<MLModelUpdateResponse> actionListener) {
        MLModelUpdateRequest mlModelGetRequest = MLModelUpdateRequest.fromActionRequest(request);
        String modelId = mlModelGetRequest.getModelId();
        MLUpdateModelInput updateModelInput = mlModelGetRequest.getUpdateModelInput();


        try (ThreadContext.StoredContext context = client.threadPool().getThreadContext().stashContext()) {
            GetRequest getRequest = new GetRequest(ML_MODEL_INDEX).id(modelId);
            client.get(getRequest, ActionListener.wrap(r -> {
                log.debug("Completed Get Model Request, id:{}", modelId);

                if (r != null && r.isExists()) {
                    try (XContentParser parser = createXContentParserFromRegistry(xContentRegistry, r.getSourceAsBytesRef())) {
                        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
                        MLModel mlModel = MLModel.parse(parser);
                        if (mlModel.getModelState() == MLModelState.DEPLOYED || mlModel.getModelState() == MLModelState.PARTIALLY_DEPLOYED) {
                            actionListener.onFailure(new IllegalArgumentException("Can't update model as it's deployed"));
                            return;
                        }
                        MLModel.MLModelBuilder mlModelBuilder = mlModel.toBuilder();
                        if (updateModelInput.getModelName() != null) {
                            mlModelBuilder.name(updateModelInput.getModelName());
                        }
                        if (updateModelInput.getDescription() != null) {
                            mlModelBuilder.description(updateModelInput.getDescription());
                        }
                        if (updateModelInput.getConnector() != null) {
                            updateModelInput.getConnector().encrypt((credential) -> mlEngine.encrypt(credential));
                            if (mlModel.getConnector() == null ||
                                    (!(mlModel.getConnector() instanceof HttpConnector))
                                    || !(updateModelInput.getConnector() instanceof HttpConnector)) {
                                mlModelBuilder.connector(updateModelInput.getConnector());
                            } else {
                                HttpConnector connector = (HttpConnector)mlModel.getConnector();
                                mlModelBuilder.connector(connector.merge((HttpConnector) updateModelInput.getConnector()));
                            }

                        }
                        if (updateModelInput.getTools() != null) {
                            mlModelBuilder.tools(updateModelInput.getTools());
                        }

                        UpdateRequest updateRequest = new UpdateRequest(ML_MODEL_INDEX, modelId);
                        XContentBuilder xContentBuilder = XContentFactory.jsonBuilder();
                        mlModelBuilder.build().toXContent(xContentBuilder, ToXContent.EMPTY_PARAMS);
                        updateRequest.doc(xContentBuilder);
                        client.update(updateRequest, ActionListener.wrap(res -> {
                            actionListener.onResponse(new MLModelUpdateResponse(res.getId(), "updated"));
                        }, e -> {
                            actionListener.onFailure(e);
                        }));
                    } catch (Exception e) {
                        log.error("Failed to parse ml model" + r.getId(), e);
                        actionListener.onFailure(e);
                    }
                } else {
                    actionListener.onFailure(new MLResourceNotFoundException("Fail to find model"));
                }
            }, e -> {
                if (e instanceof IndexNotFoundException) {
                    actionListener.onFailure(new MLResourceNotFoundException("Fail to find model"));
                } else {
                    log.error("Failed to get ML model " + modelId, e);
                    actionListener.onFailure(e);
                }
            }));
        } catch (Exception e) {
            log.error("Failed to get ML model " + modelId, e);
            actionListener.onFailure(e);
        }

    }
}
