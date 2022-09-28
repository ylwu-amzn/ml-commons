package org.opensearch.ml.common.transport.custom_model.predict;

import org.opensearch.common.io.stream.Writeable;
import org.opensearch.common.xcontent.ToXContentObject;

public interface CustomModelInput extends ToXContentObject, Writeable {
}
