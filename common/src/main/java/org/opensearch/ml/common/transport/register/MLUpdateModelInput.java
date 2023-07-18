/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.register;

import lombok.Builder;
import lombok.Data;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.MLCommonsClassLoader;
import org.opensearch.ml.common.connector.Connector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;


/**
 * ML input data: algirithm name, parameters and input data set.
 */
@Data
public class MLUpdateModelInput implements ToXContentObject, Writeable {

    public static final String NAME_FIELD = "name";
    public static final String DESCRIPTION_FIELD = "description";
    public static final String CONNECTOR_FIELD = "connector";
    public static final String TOOLS_FIELD = "tools";

    private String modelName;
    private String description;

    private Connector connector;
    private List<String> tools;

    @Builder(toBuilder = true)
    public MLUpdateModelInput(String modelName,
                              String description,
                              Connector connector,
                              List<String> tools) {
        this.modelName = modelName;
        this.description = description;
        this.connector = connector;
        this.tools = tools;
    }


    public MLUpdateModelInput(StreamInput in) throws IOException {
        this.modelName = in.readString();
        this.description = in.readOptionalString();
        if (in.readBoolean()) {
            String connectorName = in.readString();
            this.connector = MLCommonsClassLoader.initConnector(connectorName, new Object[]{connectorName, in}, String.class, StreamInput.class);
        }
        if (in.readBoolean()) {
            this.tools = in.readStringList();
        }
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(modelName);
        out.writeOptionalString(description);
        if (connector != null) {
            out.writeBoolean(true);
            out.writeString(connector.getName());
            connector.writeTo(out);
        } else {
            out.writeBoolean(false);
        }
        if (tools != null && tools.size() > 0) {
            out.writeBoolean(true);
            out.writeStringCollection(tools);
        } else {
            out.writeBoolean(false);
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(NAME_FIELD, modelName);
        if (description != null) {
            builder.field(DESCRIPTION_FIELD, description);
        }
        if (connector != null) {
            builder.field(CONNECTOR_FIELD, connector);
        }
        if (tools != null) {
            builder.field(TOOLS_FIELD, tools);
        }
        builder.endObject();
        return builder;
    }


    public static MLUpdateModelInput parse(XContentParser parser, boolean deployModel) throws IOException {
        String name = null;
        String description = null;
        Connector connector = null;
        List<String> tools = null;

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case NAME_FIELD:
                    name = parser.text();
                    break;
                case DESCRIPTION_FIELD:
                    description = parser.text();
                    break;
                case CONNECTOR_FIELD:
                    parser.nextToken();
                    String connectorName = parser.currentName();
                    parser.nextToken();
                    connector = MLCommonsClassLoader.initConnector(connectorName, new Object[]{connectorName, parser}, String.class, XContentParser.class);
                    parser.nextToken();
                    break;
                case TOOLS_FIELD:
                    tools = new ArrayList<>();
                    ensureExpectedToken(XContentParser.Token.START_ARRAY, parser.currentToken(), parser);
                    while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
                        tools.add(parser.text());
                    }
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return new MLUpdateModelInput(name, description, connector, tools);
    }
}
