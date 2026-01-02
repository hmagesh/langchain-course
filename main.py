from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults


# Create the Tavily search tool instance
tavily_tool = TavilySearchResults(max_results=3)

llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
tools = [tavily_tool]

# Create a custom prompt that FORCES the agent to use the search tool
prompt = PromptTemplate.from_template(
    """IMPORTANT: You MUST use the search tool for EVERY question, even if you think you know the answer.
Never answer directly from your knowledge. Always search first.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: I must use the search tool to find current information
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember: ALWAYS use the search tool first!

Question: {input}
Thought:{agent_scratchpad}"""
)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True
)    

def main():
    print("Hello from langchain-course!")
    
    # Now the agent will ALWAYS use the search tool, even for simple questions
    question = "what is the capital of Japan?"
    print(f"\n❓ Question: {question}\n")
    
    result = agent_executor.invoke({"input": question})
    
    # Show the intermediate steps to see how the agent used the tool
    print(f"\n{'='*60}")
    print(f"✅ Final Answer: {result['output']}")
    print(f"{'='*60}")
    print(f"\n📊 Intermediate Steps: {len(result['intermediate_steps'])} step(s) taken")   

if __name__ == "__main__":
    main()
