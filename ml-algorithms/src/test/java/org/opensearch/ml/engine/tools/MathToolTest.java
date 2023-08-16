package org.opensearch.ml.engine.tools;

import org.junit.Before;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnitRunner;
import org.opensearch.client.Client;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.script.ScriptService;

@RunWith(MockitoJUnitRunner.class)
public class MathToolTest {

    @Mock
    private ScriptService scriptService;

    @Mock
    private Client client;

    private NamedXContentRegistry xContentRegistry;


    @Before
    public void setUp() {
        xContentRegistry = NamedXContentRegistry.EMPTY;
    }


}
