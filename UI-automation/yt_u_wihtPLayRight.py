import asyncio
from typing import Any, Dict, List, TypedDict
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt.tool_executor import ToolExecutor
import nest_asyncio

nest_asyncio.apply()

# Create async browser
async_browser = create_async_playwright_browser()

# Create toolkit and get tools
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
all_tools = toolkit.get_tools()

# Select the specific tools we want to use
tools = [
    tool for tool in all_tools 
    if tool.name in ["navigate", "current_webpage", "click", "get_elements"]
]

# Define the state schema
class AgentState(BaseModel):
    task: str
    tools: List[BaseTool] = Field(default_factory=lambda: tools)
    memory: ConversationBufferMemory = Field(default_factory=ConversationBufferMemory)
    messages: List[str] = Field(default_factory=list)

class GraphState(TypedDict):
    agent_state: AgentState
    next_action: str

async def agent_node(state: GraphState) -> GraphState:
    agent_state = state['agent_state']
    
    llm = ChatOpenAI(temperature=0)
    
    agent = initialize_agent(
        tools=agent_state.tools,
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=agent_state.memory,
    )
    
    # Add a system message to guide the agent
    system_message = """You are an AI assistant that uses tools to complete tasks. Always use the following format:

Action: [tool name]
Action Input: [tool input]

Available tools: navigate_browser, search"""
    
    result = await agent.arun(system_message + "\n\nTask: " + agent_state.task)
    
    agent_state.messages.append(str(result))
    
    if "task completed" in result.lower():
        return {"agent_state": agent_state, "next_action": "end"}
    else:
        return {"agent_state": agent_state, "next_action": "continue"}

async def tool_executor_node(state: GraphState) -> GraphState:
    agent_state = state['agent_state']
    tool_executor = ToolExecutor(agent_state.tools)
    
    last_message = agent_state.messages[-1]
    print(f"Debug - Last message: {last_message}")  # Debug print
    
    # Check if the last message is a valid tool call
    if "Action:" in last_message and "Action Input:" in last_message:
        # Parse the last_message to extract the tool name and input
        tool_parts = last_message.split('Action:')[-1].strip().split('\nAction Input:')
        tool_name = tool_parts[0].strip()
        tool_input = tool_parts[1].strip() if len(tool_parts) > 1 else ""
        
        print(f"Debug - Tool Name: {tool_name}")  # Debug print
        print(f"Debug - Tool Input: {tool_input}")  # Debug print
        
        try:
            # Check if the tool exists
            if tool_name in [tool.name for tool in agent_state.tools]:
                # Execute the tool
                result = await tool_executor.ainvoke({tool_name: tool_input})
                agent_state.memory.save_context({"human": f"{tool_name}: {tool_input}"}, {"ai": str(result)})
                agent_state.messages.append(f"Executed {tool_name}: {result}")
            else:
                error_message = f"Tool '{tool_name}' not found in available tools."
                print(error_message)
                agent_state.messages.append(error_message)
        except Exception as e:
            print(f"Error executing tool {tool_name}: {str(e)}")
            agent_state.messages.append(f"Error executing {tool_name}: {str(e)}")
    else:
        print("Debug - Last message is not a valid tool call")
        # Try to interpret the message and use the appropriate tool
        if "navigate" in last_message.lower():
            tool_name = "navigate_browser"
            tool_input = {"url": "https://www.google.com"}
        elif "search" in last_message.lower():
            tool_name = "search"
            search_query = "distance between sun and earth in kilometers"
            tool_input = {"query": search_query}
        else:
            tool_name = None
            tool_input = None
        
        if tool_name and tool_input:
            try:
                result = await tool_executor.ainvoke({tool_name: tool_input})
                agent_state.memory.save_context({"human": f"{tool_name}: {tool_input}"}, {"ai": str(result)})
                agent_state.messages.append(f"Executed {tool_name}: {result}")
            except Exception as e:
                print(f"Error executing tool {tool_name}: {str(e)}")
                agent_state.messages.append(f"Error executing {tool_name}: {str(e)}")
        else:
            agent_state.messages.append("Could not determine appropriate tool from the message.")
    
    return {"agent_state": agent_state, "next_action": "continue"}

# Function to determine the next action
def should_continue(state: GraphState) -> str:
    return "end" if state["next_action"] == "end" else "continue"

# Create the graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tool_executor", tool_executor_node)

# Set the entry point
workflow.add_edge(START, "agent")

# Add conditional edges from agent node
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tool_executor",
        "end": END,
    },
)

# Add edge from tool_executor back to agent
workflow.add_edge("tool_executor", "agent")

# Compile the graph
app = workflow.compile()

# Function to run the agent
async def run_agent(task: str) -> Dict[str, Any]:
    initial_state = GraphState(
        agent_state=AgentState(task=task, tools=tools),
        next_action="continue"
    )
    result = await app.ainvoke(initial_state)
    return {
        "messages": result["agent_state"].messages,
        "memory": result["agent_state"].memory.load_memory_variables({})["history"],
    }

# Example usage
async def main():
    print("Available tools:", [tool.name for tool in tools])  # Debug print
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