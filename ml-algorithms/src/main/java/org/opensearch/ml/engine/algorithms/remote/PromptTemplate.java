package org.opensearch.ml.engine.algorithms.remote;

public class PromptTemplate {

    public static final String AGENT_TEMPLATE = "Answer the following questions as best you can. You have access to the following tools:\n\n" +
            "${tools}\n" +
            "${opensearch_indices}" +
            "Use the following format (Do NOT add sequence number after Action and Action Input):\n\n" +
            "Question: the input question you must answer\n" +
            "Thought: you should always think about what to do\n" +
            "Action: the action to take, should be one of [${tool_names}]\n" +
            "Action Input: the input to the action\n" +
            "Observation: the result of the action\n" +
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n" +
            "Thought: I now know the final answer\n" +
            "Final Answer: the final answer to the original input question\n\n" +
            "${examples}" +
            "Begin!\n\n" +
            "Question: ${parameters.question}\n" +
            "Thought: ${agent_scratchpad}\n";

    public static final String AGENT_TEMPLATE_WITH_CONTEXT = "Answer the following questions as best you can. Always try to answer question based on Context or Chat History first.\nYou have access to the following tools:\n\n" +
            "${tools}\n" +
            "${opensearch_indices}" +
            "Use the following format (Do NOT add sequence number after Action and Action Input):\n\n" +
            "Question: the input question you must answer\n" +
            "Thought: you should always think about what to do. If you can find final answer from given Context, just give the final answer, NO need to run Action any more,\n" +
            "Action: the action to take, should be one of [${tool_names}]\n" +
            "Action Input: the input to the action\n" +
            "Observation: the result of the action\n" +
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n" +
            "Thought: I now know the final answer\n" +
            "Final Answer: the final answer to the original input question\n\n" +
            "${examples}" +
            "${context}" +
            "${chat_history}" +
            "Begin!\n\n" +
            "Question: ${parameters.question}\n" +
            "Thought: ${agent_scratchpad}\n";
}
