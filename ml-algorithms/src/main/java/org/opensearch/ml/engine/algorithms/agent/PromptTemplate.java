package org.opensearch.ml.engine.algorithms.agent;

import static org.opensearch.ml.engine.algorithms.agent.MLCoTAgentRunner.CHAT_HISTORY;
import static org.opensearch.ml.engine.algorithms.agent.MLCoTAgentRunner.CONTEXT;
import static org.opensearch.ml.engine.algorithms.agent.MLCoTAgentRunner.EXAMPLES;
import static org.opensearch.ml.engine.algorithms.agent.MLCoTAgentRunner.OS_INDICES;
import static org.opensearch.ml.engine.algorithms.agent.MLCoTAgentRunner.PROMPT_PREFIX;
import static org.opensearch.ml.engine.algorithms.agent.MLCoTAgentRunner.PROMPT_SUFFIX;
import static org.opensearch.ml.engine.algorithms.agent.MLCoTAgentRunner.QUESTION;
import static org.opensearch.ml.engine.algorithms.agent.MLCoTAgentRunner.SCRATCHPAD;
import static org.opensearch.ml.engine.algorithms.agent.MLCoTAgentRunner.TOOL_DESCRIPTIONS;
import static org.opensearch.ml.engine.algorithms.agent.MLCoTAgentRunner.TOOL_NAMES;

public class PromptTemplate {


    public static final String AGENT_TEMPLATE_WITH_CONTEXT = "${parameters." + PROMPT_PREFIX + "}\n" +
            "Answer the following questions as best you can. Always try to answer question based on Context or Chat History first. If you find answer in Context or Chat History, no need to run action any more, just return the final answer.\n\n" +
            "${parameters." + CONTEXT + "}\n" +
            "${parameters." + CHAT_HISTORY + "}\n" +
            "${parameters." + TOOL_DESCRIPTIONS + "}\n" +
            "${parameters." + OS_INDICES + "}\n" +
            "${parameters." + EXAMPLES + "}\n" +
            "Use the style of Thought, Action, Observation as demonstrated below to answer the questions (Do NOT add sequence number after Action and Action Input):\n\n" +
            "Question: the input question you must answer\n" +
            "Thought: you should always think about what to do. If you can find final answer from given Context, just give the final answer, NO need to run Action any more,\n" +
            "Action: the action to take, should be one of these tool names: [${parameters." + TOOL_NAMES + "}]. Don't any any words or punctuation before or after. \n" +
            "Action Input: the input to the action\n" +
            "Observation: the result of the action\n" +
            "... (this Thought/Action/Action Input/Observation can repeat N times)\n" +
            "Thought: I now know the final answer\n" +
            "Final Answer: the final answer to the original input question\n\n" +
            "Begin!\n\n" +
            "Question: ${parameters." + QUESTION + "}\n" +
            "Thought: ${parameters." + SCRATCHPAD + "}\n" +
            "${parameters." + PROMPT_SUFFIX + "}\n" ;

}
