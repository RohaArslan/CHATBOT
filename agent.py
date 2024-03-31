from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from llm import llm
from langchain.tools import Tool
from Tools.cypher import cypher_qa

from langchain.prompts import PromptTemplate
# Include the LLM from a previous lesson
from langchain.prompts import PromptTemplate
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=False
        ),
    Tool.from_function(
        name="Graph Cypher QA Chain",  # (1)
        description="",  # (2)
        func=cypher_qa,  # (3)
        return_direct=False
    ),

]
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)


agent_prompt = PromptTemplate.from_template("""
You are a twitter expert providing assistance on different tweets.
Your goal is to gather as much relevant information as possible and provide helpful insights to provide information about tweets.
Only answer questions related to the twitter and tweets.
Do not answer any questions that do not relate to twitter, tweets, retweets, and followers
Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

Fine Tuning:

If user writes "post" it means "tweet".

To use a tool, please use the following format:
TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
    )
def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = agent_executor.invoke({"input": prompt})
    final_response = response.get('output', 'No response')

    return response['output']