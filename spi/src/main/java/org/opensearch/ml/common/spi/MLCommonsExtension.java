package org.opensearch.ml.common.spi;

import org.opensearch.ml.common.spi.tools.Tool;

import java.util.List;

public interface MLCommonsExtension {

    List<Tool> getTools();
}
