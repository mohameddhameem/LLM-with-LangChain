import json
import asyncio
from typing import Any, Dict, List, TypedDict
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
import nest_asyncio

nest_asyncio.apply()

# Create async browser
async_browser = create_async_playwright_browser()

# Create toolkit and get tools
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()

# Define the state schema
class AgentState(BaseModel):
    task: str
    tools: List[BaseTool] = Field(default_factory=lambda: toolkit.get_tools())
    memory: ConversationBufferMemory = Field(default_factory=ConversationBufferMemory)
    messages: List[str] = Field(default_factory=list)

class GraphState(TypedDict):
    agent_state: AgentState
    tool_invocations: List[ToolInvocation]

async def agent(state: GraphState) -> GraphState:
    agent_state = state['agent_state']
    llm = ChatOpenAI(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that helps with web automation tasks. Use the provided tools to complete the task. The available tools are: navigate_browser, click_element, extract_text, extract_hyperlinks, get_elements, and current_webpage. Respond with the tool name and input in JSON format. For example: {\"action\": \"navigate_browser\", \"action_input\": {\"url\": \"https://www.google.com\"}}. If you believe the task is complete, respond with COMPLETE: reason."),
        ("human", "{task}"),
        ("human", "Current conversation:\n{memory}\n\nHuman: What should I do next?"),
        ("ai", "{agent_action}")  # Changed from "action" to "agent_action"
    ])
    chain = prompt | llm
    
    agent_action = ""
    
    while True:
        result = await chain.ainvoke({
            "task": agent_state.task,
            "memory": agent_state.memory.load_memory_variables({})["history"],
            "agent_action": agent_action  # Changed from "action" to "agent_action"
        })
        
        content = result.content

        print(f"Agent received task: {agent_state.task}")
        print(f"Agent action: {content}")
        
        if content.strip().upper().startswith("COMPLETE:"):
            print("Agent completed the task")
            return {"agent_state": agent_state, "tool_invocations": [END]}
        
        try:
            action_data = json.loads(content)
            tool_name = action_data["action"]
            tool_input = action_data["action_input"]
            
            tool_invocation = ToolInvocation(tool=tool_name, tool_input=tool_input)
            
            agent_state.messages.append(str(result))
            
            return {
                "agent_state": agent_state,
                "tool_invocations": [tool_invocation]
            }
        except json.JSONDecodeError:
            agent_action = "The previous response was not in the correct format. Please provide a valid JSON response with 'action' and 'action_input' fields."

async def tool_executor(state: GraphState) -> GraphState:
    agent_state = state['agent_state']
    tool_invocation = state['tool_invocations'][0]
    tool_executor = ToolExecutor(tools)
    try:
        result = await tool_executor.ainvoke(tool_invocation)
        agent_state.memory.save_context({"human": tool_invocation.tool}, {"ai": str(result)})
        agent_state.messages.append(f"Executed {tool_invocation.tool}: {result}")
    except Exception as e:
        print(f"Error executing tool {tool_invocation.tool}: {str(e)}")
        agent_state.messages.append(f"Error executing {tool_invocation.tool}: {str(e)}")
    return {"agent_state": agent_state, "tool_invocations": []}

# Create the graph
workflow = StateGraph(GraphState)
workflow.add_node("agent", agent)
workflow.add_node("tool_executor", tool_executor)
workflow.add_edge("agent", "tool_executor")
workflow.add_edge("tool_executor", "agent")
workflow.set_entry_point("agent")

# Compile the graph
app = workflow.compile()

# Function to run the agent
async def run_agent(task: str) -> Dict[str, Any]:
    initial_state = GraphState(
        agent_state=AgentState(task=task, tools=tools),
        tool_invocations=[]
    )
    result = await app.ainvoke(initial_state)
    if result.get("tool_invocations") == [END]:
        print("Task completed successfully")
    return {
        "messages": result["agent_state"].messages,
        "memory": result["agent_state"].memory.load_memory_variables({})["history"],
    }

# Example usage
async def main():
    task = "Navigate to google.com and search for the distance between sun and earth in kilometers"
    result = await run_agent(task)

    print("Agent messages:")
    for message in result["messages"]:
        print(message)

    print("\nAgent memory:")
    print(result["memory"])

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())