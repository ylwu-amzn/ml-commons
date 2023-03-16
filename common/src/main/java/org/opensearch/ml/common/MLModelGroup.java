/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common;

import lombok.Builder;
import lombok.Getter;
import org.opensearch.common.io.stream.StreamInput;
import org.opensearch.common.io.stream.StreamOutput;
import org.opensearch.common.xcontent.ToXContentObject;
import org.opensearch.common.xcontent.XContentBuilder;

import java.io.IOException;
import java.util.List;

@Getter
public class MLModelGroup implements ToXContentObject {
    public static final String MODEL_GROUP_NAME_FIELD = "name";
    // We use int type for version in first release 1.3. In 2.4, we changed to
    // use String type for version. Keep this old version field for old models.
    public static final String DESCRIPTION_FIELD = "description";
    public static final String LATEST_VERSION_FIELD = "latest_version";
    //SHA256 hash value of model content.

    public static final String TAGS_FIELD = "tags";
    public static final String MODEL_IDS_FIELD = "model_ids";
    //TODO: add created time, updated time,
    private String name;
    private String description;
    private List<String> tags;
    private List<String> models;
    private int latestVersion = 0;


    @Builder(toBuilder = true)
    public MLModelGroup(String name, String description, List<String> tags, List<String> models, int latestVersion) {
        this.name = name;
        this.description = description;
        this.tags = tags;
        this.models = models;
        this.latestVersion = latestVersion;
    }


    public MLModelGroup(StreamInput input) throws IOException{
        name = input.readString();
        description = input.readOptionalString();
        latestVersion = input.readInt();
        if (input.readBoolean()) {
            this.tags = input.readStringList();
        } else {
            this.tags = null;
        }
        if (input.readBoolean()) {
            this.models = input.readStringList();
        } else {
            this.models = null;
        }
    }

    public void writeTo(StreamOutput out) throws IOException {
        out.writeString(name);
        out.writeOptionalString(description);
        out.writeInt(latestVersion);
        if (tags != null) {
            out.writeBoolean(true);
            out.writeStringCollection(tags);
        } else {
            out.writeBoolean(false);
        }
        if (models != null) {
            out.writeBoolean(true);
            out.writeStringCollection(models);
        } else {
            out.writeBoolean(false);
        }
    }

    @Override
    public XContentBuilder toXContent(XContentBuilder builder, Params params) throws IOException {
        builder.startObject();
        builder.field(MODEL_GROUP_NAME_FIELD, name);
        builder.field(LATEST_VERSION_FIELD, latestVersion);
        if (description != null) {
            builder.field(DESCRIPTION_FIELD, description);
        }
        if (tags != null && tags.size() > 0) {
            builder.field(TAGS_FIELD, tags);
        }
        if (models != null && models.size() > 0) {
            builder.field(MODEL_IDS_FIELD, models);
        }
        builder.endObject();
        return builder;
    }


    public static MLModelGroup fromStream(StreamInput in) throws IOException {
        MLModelGroup mlModel = new MLModelGroup(in);
        return mlModel;
    }
}
