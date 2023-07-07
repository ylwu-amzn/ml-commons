/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.connector;

import lombok.Builder;
import lombok.Data;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.AccessMode;
import org.opensearch.ml.common.connector.ConnectorAction;
import org.opensearch.ml.common.connector.template.ConnectorTemplate;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;

@Data
public class MLCreateConnectorInput implements ToXContentObject, Writeable {
    public static final String CONNECTOR_NAME_FIELD = "name";
    public static final String CONNECTOR_DESCRIPTION_FIELD = "description";
    public static final String CONNECTOR_VERSION_FIELD = "version";
    public static final String CONNECTOR_PROTOCOL_FIELD = "protocol";

    public static final String CONNECTOR_PARAMETERS_FIELD = "parameters";
    public static final String CONNECTOR_CREDENTIAL_FIELD = "credential";
    public static final String CONNECTOR_ACTIONS_FIELD = "actions";

    public static final String BACKEND_ROLES_FIELD = "backend_roles";
    public static final String ADD_ALL_BACKEND_ROLES_FIELD = "add_all_backend_roles";
    public static final String OWNER_FIELD = "owner";
    public static final String ACCESS_MODE_FIELD = "access_mode";

    public static final String DRY_RUN_CONNECTOR_NAME = "dryRunConnector";

    private String name;
    private String description;
    private String version;
    private String protocol;
    private Map<String, String> parameters;
    private Map<String, String> credential;
    private List<ConnectorAction> actions;
    private List<String> backendRoles;
    private Boolean addAllBackendRoles;
    private AccessMode access;

    @Builder(toBuilder = true)
    public MLCreateConnectorInput(String name,
                                  String description,
                                  String version,
                                  String protocol,
                                  Map<String, String> parameters,
                                  Map<String, String> credential,
                                  List<ConnectorAction> actions,
                                  List<String> backendRoles,
                                  Boolean addAllBackendRoles,
                                  AccessMode access
                                  ) {
        this.name = name;
        this.description = description;
        this.version = version;
        this.protocol = protocol;
        this.parameters = parameters;
        this.credential = credential;
        this.actions = actions;
        this.backendRoles = backendRoles;
        this.addAllBackendRoles = addAllBackendRoles;
        this.access = access;
    }

    public static MLCreateConnectorInput parse(XContentParser parser) throws IOException {
        String name = null;
        String description = null;
        String version = null;
        String protocol = null;
        Map<String, String> parameters = new HashMap<>();
        Map<String, String> credential = new HashMap<>();
        List<ConnectorAction> actions = null;
        List<String> backendRoles = null;
        Boolean addAllBackendRoles = null;
        AccessMode access = null;

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case CONNECTOR_NAME_FIELD:
                    name = parser.text();
                    break;
                case CONNECTOR_DESCRIPTION_FIELD:
                    description = parser.text();
                    break;
                case CONNECTOR_VERSION_FIELD:
                    version = parser.text();
                    break;
                case CONNECTOR_PROTOCOL_FIELD:
                    protocol = parser.text();
                    break;
                case CONNECTOR_PARAMETERS_FIELD:
                    parameters = parser.mapStrings();
                    break;
                case CONNECTOR_CREDENTIAL_FIELD:
                    credential = parser.mapStrings();
                    break;
                case CONNECTOR_ACTIONS_FIELD:
                    actions = new ArrayList<>();
                    ensureExpectedToken(XContentParser.Token.START_ARRAY, parser.currentToken(), parser);
                    while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
                        actions.add(ConnectorAction.parse(parser));
                    }
                    break;
                case BACKEND_ROLES_FIELD:
                    ensureExpectedToken(XContentParser.Token.START_ARRAY, parser.currentToken(), parser);
                    backendRoles = new ArrayList<>();
                    while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
                        backendRoles.add(parser.text());
                    }
                    break;
                case ADD_ALL_BACKEND_ROLES_FIELD:
                    addAllBackendRoles = parser.booleanValue();
                    break;
                case ACCESS_MODE_FIELD:
                    access = AccessMode.from(parser.text());
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return new MLCreateConnectorInput(name, description, version, protocol, parameters, credential, actions, backendRoles, addAllBackendRoles, access);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        if (name != null) {
            builder.field(CONNECTOR_NAME_FIELD, name);
        }
        if (description != null) {
            builder.field(CONNECTOR_DESCRIPTION_FIELD, description);
        }
        if (version != null) {
            builder.field(CONNECTOR_VERSION_FIELD, version);
        }
        if (protocol != null) {
            builder.field(CONNECTOR_PROTOCOL_FIELD, protocol);
        }
        if (parameters != null) {
            builder.field(CONNECTOR_PARAMETERS_FIELD, parameters);
        }
        if (credential != null) {
            builder.field(CONNECTOR_CREDENTIAL_FIELD, credential);
        }
        if (actions != null) {
            builder.field(CONNECTOR_ACTIONS_FIELD, actions);
        }
        if (backendRoles != null) {
            builder.field(BACKEND_ROLES_FIELD, backendRoles);
        }
        if (addAllBackendRoles != null) {
            builder.field(ADD_ALL_BACKEND_ROLES_FIELD, addAllBackendRoles);
        }
        if (access != null) {
            builder.field(ACCESS_MODE_FIELD, access);
        }
        builder.endObject();
        return builder;
    }

    @Override
    public void writeTo(StreamOutput output) throws IOException {
        output.writeString(name);
        output.writeString(description);
        output.writeString(version);
        output.writeString(protocol);
        if (parameters != null) {
            output.writeBoolean(true);
            output.writeMap(parameters, StreamOutput::writeString, StreamOutput::writeString);
        } else {
            output.writeBoolean(false);
        }
        if (credential != null) {
            output.writeBoolean(true);
            output.writeMap(credential, StreamOutput::writeString, StreamOutput::writeString);
        } else {
            output.writeBoolean(false);
        }
        if (actions != null) {
            output.writeBoolean(true);
            output.writeInt(actions.size());
            for (ConnectorAction action : actions) {
                action.writeTo(output);
            }
        } else {
            output.writeBoolean(false);
        }
        if (backendRoles != null) {
            output.writeBoolean(true);
            output.writeOptionalStringCollection(backendRoles);
        } else {
            output.writeBoolean(false);
        }
        if (addAllBackendRoles != null) {
            output.writeBoolean(addAllBackendRoles);
        }
        if (access != null) {
            output.writeBoolean(true);
            output.writeEnum(access);
        } else {
            output.writeBoolean(false);
        }
    }

    public MLCreateConnectorInput(StreamInput input) throws IOException {
        name = input.readString();
        description = input.readString();
        version = input.readString();
        protocol = input.readString();
        if (input.readBoolean()) {
            parameters = input.readMap(s -> s.readString(), s -> s.readString());
        }
        if (input.readBoolean()) {
            credential = input.readMap(s -> s.readString(), s-> s.readString());
        }
        if (input.readBoolean()) {
            actions = new ArrayList<>();
            int size = input.readInt();
            for (int i = 0; i < size; i++) {
                actions.add(new ConnectorAction(input));
            }
        }
        if (input.readBoolean()) {
            this.backendRoles = input.readList(StreamInput::readString);
        }
        this.addAllBackendRoles = input.readOptionalBoolean();
        if (input.readBoolean()) {
            this.access = input.readEnum(AccessMode.class);
        }
    }
}
