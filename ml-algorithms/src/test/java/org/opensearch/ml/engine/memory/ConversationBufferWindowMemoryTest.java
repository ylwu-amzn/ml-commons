package org.opensearch.ml.engine.memory;

import org.junit.Assert;
import org.junit.Test;
import org.opensearch.ml.common.spi.memory.Message;

public class ConversationBufferWindowMemoryTest {

    @Test
    public void onlyStoreLatestMessages() {
        ConversationBufferWindowMemory memory = new ConversationBufferWindowMemory();
        memory.save("id1", BaseMessage.builder().type("Human").content("hello").build());
        memory.save("id1", BaseMessage.builder().type("AI").content("How are your? What I can do for you").build());
        memory.save("id1", BaseMessage.builder().type("Human").content("What's your name").build());
        memory.save("id1", BaseMessage.builder().type("AI").content("My name is xyz. What's your name").build());
        memory.save("id1", BaseMessage.builder().type("Human").content("If your name is xyz, my name is abc").build());
        memory.save("id1", BaseMessage.builder().type("AI").content("Is that a joke").build());
        memory.save("id1", BaseMessage.builder().type("Human").content("You can guess").build());
        memory.save("id1", BaseMessage.builder().type("AI").content("That sounds a joke").build());
        Message[] messages = memory.getMessages("id1");
        int length = messages.length;
        Assert.assertEquals(memory.getWindowSize(), length);
        Assert.assertEquals("AI: My name is xyz. What's your name", messages[0].toString());
    }
}
