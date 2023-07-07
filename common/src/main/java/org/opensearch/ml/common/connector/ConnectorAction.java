/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.connector;

import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.io.stream.Writeable;
import org.opensearch.core.xcontent.ToXContentObject;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;

import static org.opensearch.common.xcontent.XContentParserUtils.ensureExpectedToken;

@Getter
@EqualsAndHashCode
public class ConnectorAction implements ToXContentObject, Writeable {

    public static final String ACTION_TYPE_FIELD = "action_type";
    public static final String METHOD_FIELD = "method";
    public static final String URL_FIELD = "url";
    public static final String HEADERS_FIELD = "headers";
    public static final String REQUEST_BODY_FIELD = "request_body";
    //TODO: add process function to action level
    public static final String PRE_PROCESS_FUNCTION_FIELD = "pre_process_function";
    public static final String POST_PROCESS_FUNCTION_FIELD = "post_process_function";

    private ActionType actionType;
    private String method;
    private String url;
    private Map<String, String> headers;
    private String requestBody;

    @Builder(toBuilder = true)
    public ConnectorAction(
        ActionType actionType,
        String method,
        String url,
        Map<String, String> headers,
        String requestBody
    ) {
        if (actionType == null) {
            throw new IllegalArgumentException("action type can't null");
        }
        if (url == null) {
            throw new IllegalArgumentException("url can't null");
        }
        if (method == null) {
            throw new IllegalArgumentException("method can't null");
        }
        this.actionType = actionType;
        this.method = method;
        this.url = url;
        this.headers = headers;
        this.requestBody = requestBody;
    }

    public ConnectorAction(StreamInput input) throws IOException {
        this.actionType = input.readEnum(ActionType.class);
        this.method = input.readString();
        this.url = input.readString();
        if (input.readBoolean()) {
            this.headers = input.readMap(StreamInput::readString, StreamInput::readString);
        }
        this.requestBody = input.readOptionalString();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeEnum(actionType);
        out.writeString(method);
        out.writeString(url);
        if (headers != null) {
            out.writeBoolean(true);
            out.writeMap(headers, StreamOutput::writeString, StreamOutput::writeString);
        }
        out.writeOptionalString(requestBody);
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder xContentBuilder, Params params) throws IOException {
        XContentBuilder builder = xContentBuilder.startObject();
        if (actionType != null) {
            builder.field(ACTION_TYPE_FIELD, actionType);
        }
        if (method != null) {
            builder.field(METHOD_FIELD, method);
        }
        if (url != null) {
            builder.field(URL_FIELD, url);
        }
        if (headers != null) {
            builder.field(HEADERS_FIELD, headers);
        }
        if (requestBody != null) {
            builder.field(REQUEST_BODY_FIELD, requestBody);
        }
        return builder.endObject();
    }

    public static ConnectorAction fromStream(StreamInput in) throws IOException {
        ConnectorAction action = new ConnectorAction(in);
        return action;
    }

    public static ConnectorAction parse(XContentParser parser) throws IOException {
        ActionType actionType = null;
        String method = null;
        String url = null;
        Map<String, String> headers = null;
        String requestBody = null;

        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();

            switch (fieldName) {
                case ACTION_TYPE_FIELD:
                    actionType = ActionType.valueOf(parser.text().toUpperCase(Locale.ROOT));
                    break;
                case METHOD_FIELD:
                    method = parser.text();
                    break;
                case URL_FIELD:
                    url = parser.text();
                    break;
                case HEADERS_FIELD:
                    headers = parser.mapStrings();
                    break;
                case REQUEST_BODY_FIELD:
                    requestBody = parser.text();
                    break;
                default:
                    parser.skipChildren();
                    break;
            }
        }
        return ConnectorAction.builder()
                .actionType(actionType)
                .method(method)
                .url(url)
                .headers(headers)
                .requestBody(requestBody)
                .build();
    }

    public enum ActionType {
        PREDICT
    }
}
