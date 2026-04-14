/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.memorycontainer.memory;

import static org.opensearch.common.xcontent.json.JsonXContent.jsonXContent;
import static org.opensearch.core.xcontent.XContentParserUtils.ensureExpectedToken;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.DEFAULT_LLM_RESULT_PATH;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.ENTITY_EXTRACTION_PROMPT;
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.RELATIONSHIP_EXTRACTION_PROMPT;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opensearch.OpenSearchException;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.memorycontainer.MemoryConfiguration;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.transport.memorycontainer.memory.MessageInput;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.common.utils.StringUtils;
import org.opensearch.ml.engine.processor.MLProcessorType;
import org.opensearch.ml.engine.processor.ProcessorChain;
import org.opensearch.transport.client.Client;

import com.jayway.jsonpath.JsonPath;

import lombok.extern.log4j.Log4j2;

/**
 * Service for processing conversations to extract entities and relationships using LLM
 */
@Log4j2
public class GraphProcessingService {

    private final Client client;
    private final NamedXContentRegistry xContentRegistry;
    private final ProcessorChain extractJsonProcessorChain;

    public GraphProcessingService(Client client, NamedXContentRegistry xContentRegistry) {
        this.client = client;
        this.xContentRegistry = xContentRegistry;
        List<Map<String, Object>> processorConfigs = new ArrayList<>();
        processorConfigs.add(Map.of("type", MLProcessorType.EXTRACT_JSON.getValue(), "extract_type", "object"));
        this.extractJsonProcessorChain = new ProcessorChain(processorConfigs);
    }

    /**
     * Extract entities and relationships from conversation messages using LLM
     */
    public void extractEntitiesAndRelationships(
        String tenantId,
        List<MessageInput> messages,
        MemoryConfiguration memoryConfig,
        Map<String, String> namespace,
        ActionListener<GraphExtractionResult> listener
    ) {
        String llmId = memoryConfig.getLlmId();
        if (llmId == null) {
            log.debug("No LLM configured for graph extraction, skipping");
            listener.onResponse(GraphExtractionResult.builder()
                .entities(new ArrayList<>())
                .relationships(new ArrayList<>())
                .build());
            return;
        }

        log.debug("Extracting entities and relationships using LLM model: {}", llmId);

        // First extract entities, then relationships
        extractEntitiesFromMessages(tenantId, messages, memoryConfig, ActionListener.wrap(
            entities -> {
                if (entities.isEmpty()) {
                    listener.onResponse(GraphExtractionResult.builder()
                        .entities(entities)
                        .relationships(new ArrayList<>())
                        .build());
                    return;
                }

                // Extract relationships between the found entities
                extractRelationshipsFromMessages(tenantId, messages, entities, memoryConfig, ActionListener.wrap(
                    relationships -> {
                        listener.onResponse(GraphExtractionResult.builder()
                            .entities(entities)
                            .relationships(relationships)
                            .build());
                    },
                    relationshipError -> {
                        log.warn("Failed to extract relationships, returning entities only", relationshipError);
                        listener.onResponse(GraphExtractionResult.builder()
                            .entities(entities)
                            .relationships(new ArrayList<>())
                            .build());
                    }
                ));
            },
            listener::onFailure
        ));
    }

    /**
     * Extract entities from conversation messages using LLM
     */
    private void extractEntitiesFromMessages(
        String tenantId,
        List<MessageInput> messages,
        MemoryConfiguration memoryConfig,
        ActionListener<List<ExtractedEntity>> listener
    ) {
        try {
            String conversationJson = serializeMessagesToJson(messages);
            String entityPrompt = getEntityExtractionPrompt(memoryConfig);

            Map<String, String> parameters = new HashMap<>();
            parameters.put("system_prompt", entityPrompt);
            parameters.put("user_prompt", "Extract entities from the following conversation:\n```json\n" + conversationJson + "\n```");

            callLLMForExtraction(tenantId, memoryConfig.getLlmId(), parameters, ActionListener.wrap(
                llmOutput -> {
                    try {
                        List<ExtractedEntity> entities = parseEntitiesFromLLMResponse(llmOutput);
                        // Apply entity threshold filtering
                        List<ExtractedEntity> filteredEntities = filterEntitiesByThreshold(entities, memoryConfig.getGraphEntityThreshold());
                        // Limit number of entities per extraction
                        List<ExtractedEntity> limitedEntities = limitEntities(filteredEntities, memoryConfig.getMaxEntitiesPerExtraction());
                        log.debug("Extracted {} entities from conversation (filtered: {}, limited: {})",
                                  entities.size(), filteredEntities.size(), limitedEntities.size());
                        listener.onResponse(limitedEntities);
                    } catch (Exception e) {
                        log.error("Failed to parse entities from LLM response", e);
                        listener.onFailure(e);
                    }
                },
                listener::onFailure
            ));
        } catch (Exception e) {
            log.error("Failed to extract entities from messages", e);
            listener.onFailure(e);
        }
    }

    /**
     * Extract relationships between entities from conversation messages using LLM
     */
    private void extractRelationshipsFromMessages(
        String tenantId,
        List<MessageInput> messages,
        List<ExtractedEntity> entities,
        MemoryConfiguration memoryConfig,
        ActionListener<List<ExtractedRelationship>> listener
    ) {
        try {
            String conversationJson = serializeMessagesToJson(messages);
            String relationshipPrompt = getRelationshipExtractionPrompt(memoryConfig);

            // Add entity context to help LLM understand available entities
            StringBuilder entityContext = new StringBuilder();
            entityContext.append("Available entities: ");
            for (int i = 0; i < entities.size(); i++) {
                if (i > 0) entityContext.append(", ");
                entityContext.append(entities.get(i).getName());
            }

            Map<String, String> parameters = new HashMap<>();
            parameters.put("system_prompt", relationshipPrompt);
            parameters.put("user_prompt", entityContext + "\n\nExtract relationships from the following conversation:\n```json\n" + conversationJson + "\n```");

            callLLMForExtraction(tenantId, memoryConfig.getLlmId(), parameters, ActionListener.wrap(
                llmOutput -> {
                    try {
                        List<ExtractedRelationship> relationships = parseRelationshipsFromLLMResponse(llmOutput);
                        // Apply relationship threshold filtering
                        List<ExtractedRelationship> filteredRelationships = filterRelationshipsByThreshold(relationships, memoryConfig.getGraphRelationshipThreshold());
                        log.debug("Extracted {} relationships from conversation (filtered: {})", relationships.size(), filteredRelationships.size());
                        listener.onResponse(filteredRelationships);
                    } catch (Exception e) {
                        log.error("Failed to parse relationships from LLM response", e);
                        listener.onFailure(e);
                    }
                },
                listener::onFailure
            ));
        } catch (Exception e) {
            log.error("Failed to extract relationships from messages", e);
            listener.onFailure(e);
        }
    }

    /**
     * Call LLM for extraction tasks (entities or relationships)
     */
    private void callLLMForExtraction(
        String tenantId,
        String llmId,
        Map<String, String> parameters,
        ActionListener<MLOutput> listener
    ) {
        MLInput mlInput = MLInput.builder()
            .algorithm(FunctionName.REMOTE)
            .inputDataset(RemoteInferenceInputDataSet.builder().parameters(parameters).build())
            .build();

        MLPredictionTaskRequest predictionRequest = MLPredictionTaskRequest.builder()
            .modelId(llmId)
            .mlInput(mlInput)
            .tenantId(tenantId)
            .build();

        client.execute(MLPredictionTaskAction.INSTANCE, predictionRequest, ActionListener.wrap(
            response -> {
                log.debug("Received LLM response for graph extraction");
                listener.onResponse(response.getOutput());
            },
            error -> {
                log.error("Failed to call LLM for graph extraction", error);
                // Preserve client errors (4XX) with their detailed messages
                if (error instanceof OpenSearchException) {
                    OpenSearchException osException = (OpenSearchException) error;
                    if (osException.status().getStatus() >= 400 && osException.status().getStatus() < 500) {
                        listener.onFailure(error);
                        return;
                    }
                }
                listener.onFailure(new OpenSearchStatusException("Internal server error", RestStatus.INTERNAL_SERVER_ERROR));
            }
        ));
    }

    /**
     * Parse entities from LLM response JSON
     */
    private List<ExtractedEntity> parseEntitiesFromLLMResponse(MLOutput mlOutput) {
        List<ExtractedEntity> entities = new ArrayList<>();

        if (!(mlOutput instanceof ModelTensorOutput)) {
            log.warn("Unexpected ML output type for entity extraction: {}", mlOutput != null ? mlOutput.getClass().getName() : "null");
            return entities;
        }

        ModelTensorOutput tensorOutput = (ModelTensorOutput) mlOutput;
        if (tensorOutput.getMlModelOutputs() == null || tensorOutput.getMlModelOutputs().isEmpty()) {
            log.warn("No model outputs found in LLM response for entity extraction");
            return entities;
        }

        ModelTensors modelTensors = tensorOutput.getMlModelOutputs().get(0);
        if (modelTensors.getMlModelTensors() == null || modelTensors.getMlModelTensors().isEmpty()) {
            log.warn("No model tensors found in LLM response for entity extraction");
            return entities;
        }

        for (int i = 0; i < modelTensors.getMlModelTensors().size(); i++) {
            Map<String, ?> dataMap = modelTensors.getMlModelTensors().get(i).getDataAsMap();
            try {
                Object filteredResult = JsonPath.read(dataMap, DEFAULT_LLM_RESULT_PATH);
                String llmResult = null;
                if (filteredResult != null) {
                    llmResult = StringUtils.toJson(filteredResult);
                }
                if (llmResult != null) {
                    llmResult = StringUtils.toJson(extractJsonProcessorChain.process(llmResult));
                    try (XContentParser parser = jsonXContent.createParser(xContentRegistry, LoggingDeprecationHandler.INSTANCE, llmResult)) {
                        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
                        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
                            String fieldName = parser.currentName();
                            if ("entities".equals(fieldName)) {
                                ensureExpectedToken(XContentParser.Token.START_ARRAY, parser.nextToken(), parser);
                                while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
                                    entities.add(parseEntityFromJson(parser));
                                }
                            } else {
                                parser.skipChildren();
                            }
                        }
                    }
                }
            } catch (Exception e) {
                log.error("Failed to extract entities from tensor {}", i, e);
            }
        }

        return entities;
    }

    /**
     * Parse relationships from LLM response JSON
     */
    private List<ExtractedRelationship> parseRelationshipsFromLLMResponse(MLOutput mlOutput) {
        List<ExtractedRelationship> relationships = new ArrayList<>();

        if (!(mlOutput instanceof ModelTensorOutput)) {
            log.warn("Unexpected ML output type for relationship extraction: {}", mlOutput != null ? mlOutput.getClass().getName() : "null");
            return relationships;
        }

        ModelTensorOutput tensorOutput = (ModelTensorOutput) mlOutput;
        if (tensorOutput.getMlModelOutputs() == null || tensorOutput.getMlModelOutputs().isEmpty()) {
            log.warn("No model outputs found in LLM response for relationship extraction");
            return relationships;
        }

        ModelTensors modelTensors = tensorOutput.getMlModelOutputs().get(0);
        if (modelTensors.getMlModelTensors() == null || modelTensors.getMlModelTensors().isEmpty()) {
            log.warn("No model tensors found in LLM response for relationship extraction");
            return relationships;
        }

        for (int i = 0; i < modelTensors.getMlModelTensors().size(); i++) {
            Map<String, ?> dataMap = modelTensors.getMlModelTensors().get(i).getDataAsMap();
            try {
                Object filteredResult = JsonPath.read(dataMap, DEFAULT_LLM_RESULT_PATH);
                String llmResult = null;
                if (filteredResult != null) {
                    llmResult = StringUtils.toJson(filteredResult);
                }
                if (llmResult != null) {
                    llmResult = StringUtils.toJson(extractJsonProcessorChain.process(llmResult));
                    try (XContentParser parser = jsonXContent.createParser(xContentRegistry, LoggingDeprecationHandler.INSTANCE, llmResult)) {
                        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.nextToken(), parser);
                        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
                            String fieldName = parser.currentName();
                            if ("relationships".equals(fieldName)) {
                                ensureExpectedToken(XContentParser.Token.START_ARRAY, parser.nextToken(), parser);
                                while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
                                    relationships.add(parseRelationshipFromJson(parser));
                                }
                            } else {
                                parser.skipChildren();
                            }
                        }
                    }
                }
            } catch (Exception e) {
                log.error("Failed to extract relationships from tensor {}", i, e);
            }
        }

        return relationships;
    }

    /**
     * Parse single entity from JSON object in parser
     */
    private ExtractedEntity parseEntityFromJson(XContentParser parser) throws IOException {
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);

        String name = null;
        String type = null;
        Double confidence = null;

        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case "name":
                    name = parser.text();
                    break;
                case "type":
                    type = parser.text();
                    break;
                case "confidence":
                    confidence = parser.doubleValue();
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }

        return ExtractedEntity.builder()
            .name(name)
            .type(type)
            .confidence(confidence)
            .build();
    }

    /**
     * Parse single relationship from JSON object in parser
     */
    private ExtractedRelationship parseRelationshipFromJson(XContentParser parser) throws IOException {
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);

        String source = null;
        String target = null;
        String type = null;
        Double confidence = null;

        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case "source":
                    source = parser.text();
                    break;
                case "target":
                    target = parser.text();
                    break;
                case "type":
                    type = parser.text();
                    break;
                case "confidence":
                    confidence = parser.doubleValue();
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }

        return ExtractedRelationship.builder()
            .sourceEntity(source)
            .targetEntity(target)
            .type(type)
            .confidence(confidence)
            .build();
    }

    /**
     * Filter entities by confidence threshold
     */
    private List<ExtractedEntity> filterEntitiesByThreshold(List<ExtractedEntity> entities, Double threshold) {
        if (threshold == null) {
            return entities;
        }

        return entities.stream()
            .filter(entity -> entity.getConfidence() == null || entity.getConfidence() >= threshold)
            .toList();
    }

    /**
     * Filter relationships by confidence threshold
     */
    private List<ExtractedRelationship> filterRelationshipsByThreshold(List<ExtractedRelationship> relationships, Double threshold) {
        if (threshold == null) {
            return relationships;
        }

        return relationships.stream()
            .filter(relationship -> relationship.getConfidence() == null || relationship.getConfidence() >= threshold)
            .toList();
    }

    /**
     * Limit number of entities per extraction
     */
    private List<ExtractedEntity> limitEntities(List<ExtractedEntity> entities, Integer maxEntities) {
        if (maxEntities == null || entities.size() <= maxEntities) {
            return entities;
        }

        // Sort by confidence (descending) and take top N
        return entities.stream()
            .sorted((a, b) -> {
                if (a.getConfidence() == null && b.getConfidence() == null) return 0;
                if (a.getConfidence() == null) return 1;
                if (b.getConfidence() == null) return -1;
                return Double.compare(b.getConfidence(), a.getConfidence());
            })
            .limit(maxEntities)
            .toList();
    }

    /**
     * Get entity extraction prompt (custom or default)
     */
    private String getEntityExtractionPrompt(MemoryConfiguration config) {
        String customPrompt = config.getCustomEntityExtractionPrompt();
        return (customPrompt != null && !customPrompt.isBlank()) ? customPrompt : ENTITY_EXTRACTION_PROMPT;
    }

    /**
     * Get relationship extraction prompt (custom or default)
     */
    private String getRelationshipExtractionPrompt(MemoryConfiguration config) {
        String customPrompt = config.getCustomRelationshipExtractionPrompt();
        return (customPrompt != null && !customPrompt.isBlank()) ? customPrompt : RELATIONSHIP_EXTRACTION_PROMPT;
    }

    /**
     * Serialize messages to JSON format for LLM processing
     */
    private String serializeMessagesToJson(List<MessageInput> messages) throws IOException {
        // Same format as MemoryProcessingService
        List<Map<String, Object>> messageList = new ArrayList<>();
        for (MessageInput message : messages) {
            Map<String, Object> messageMap = new HashMap<>();
            messageMap.put("role", message.getRole());
            messageMap.put("content", message.getContent());
            messageList.add(messageMap);
        }
        return StringUtils.toJson(messageList);
    }
}