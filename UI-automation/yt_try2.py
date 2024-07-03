import threading
from queue import Queue
from typing import Any, TypedDict, Dict, List

from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain_community.tools.playwright.click import ClickTool
from langchain_community.tools.playwright.current_page import CurrentWebPageTool
from langchain_community.tools.playwright.extract_hyperlinks import (
    ExtractHyperlinksTool,
)
from langchain_community.tools.playwright.extract_text import ExtractTextTool
from langchain_community.tools.playwright.get_elements import GetElementsTool
from langchain_community.tools.playwright.navigate import NavigateTool
from langchain_community.tools.playwright.navigate_back import NavigateBackTool
from playwright.sync_api import sync_playwright


# Define the state schema
class AgentState(BaseModel):
    task: str
    tools: List[BaseTool]
    memory: ConversationBufferMemory = Field(default_factory=ConversationBufferMemory)
    messages: List[str] = Field(default_factory=list)


class GraphState(TypedDict):
    agent_state: AgentState
    tool_invocations: List[ToolInvocation]


# Create a queue for Playwright operations
playwright_queue = Queue()


# Playwright thread function
def playwright_thread():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        while True:
            func, args, kwargs, result_queue = playwright_queue.get()
            if func is None:
                break
            try:
                result = func(page, *args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                result_queue.put(e)
            playwright_queue.task_done()

        page.close()
        browser.close()


# Start the Playwright thread
threading.Thread(target=playwright_thread, daemon=True).start()


# Define the tools
tools = [
    NavigateTool(),
    CurrentWebPageTool(),
    ExtractHyperlinksTool(),
    ExtractTextTool(),
    GetElementsTool(),
    NavigateBackTool(),
    ClickTool(),
    ExtractTextTool()
]


# Define the state schema
class AgentState(BaseModel):
    task: str
    tools: List[BaseTool]
    memory: ConversationBufferMemory = Field(default_factory=ConversationBufferMemory)
    messages: List[str] = Field(default_factory=list)


class GraphState(TypedDict):
    agent_state: AgentState
    tool_invocations: List[ToolInvocation]


def agent(state: GraphState) -> GraphState:
    agent_state = state['agent_state']
    llm = ChatOpenAI(temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that helps with web automation tasks. Use the provided tools to complete the task. The available tools are: Navigate, Click, Type, and Extract Text. Respond with the tool name and input in the format: TOOL: tool_name, INPUT: tool_input. Ensure tool_input contains valid CSS selectors for the Click and Type tools. If you believe the task is complete, respond with COMPLETE: reason."),
        ("human", "{task}"),
        ("human", "Current conversation:\n{memory}\n\nHuman: What should I do next?"),
    ])
    chain = prompt | llm
    result = chain.invoke({
        "task": agent_state.task,
        "memory": agent_state.memory.load_memory_variables({})["history"],
    })
    
    content = result.content

    print(f"Agent received task: {agent_state.task}")
    print(f"Agent action: {content}")
    
    if content.strip().upper().startswith("COMPLETE:"):
        print("Agent completed the task")
        return {"agent_state": agent_state, "tool_invocations": [END]}
    
    if "TOOL:" in content and "INPUT:" in content:
        tool_part, input_part = content.split("INPUT:")
        tool_name = tool_part.split("TOOL:")[1].strip().strip(',')
        tool_input = input_part.strip()
        
        tool_invocation = ToolInvocation(tool=tool_name, tool_input={"input": tool_input})
    else:
        # If the output is not in the expected format, use a default tool
        tool_invocation = ToolInvocation(tool="Extract Text", tool_input={"input": "body"})
    
    agent_state.messages.append(str(result))
    
    return {
        "agent_state": agent_state,
        "tool_invocations": [tool_invocation]
    }


def tool_executor(state: GraphState) -> GraphState:
    agent_state = state['agent_state']
    tool_invocation = state['tool_invocations'][0]
    tool_executor = ToolExecutor(tools)
    try:
        result = tool_executor.invoke(tool_invocation)
        agent_state.memory.save_context({"human": tool_invocation.tool}, {"ai": str(result)})
        agent_state.messages.append(f"Executed {tool_invocation.tool}: {result}")
    except Exception as e:
        print(f"Error executing tool {tool_invocation.tool}: {str(e)}")
        agent_state.messages.append(f"Error executing {tool_invocation.tool}: {str(e)}")
    return {"agent_state": agent_state, "tool_invocations": []}

# Create the graph with increased recursion limit
workflow = StateGraph(GraphState)
workflow.add_node("agent", agent)
workflow.add_node("tool_executor", tool_executor)
workflow.add_edge("agent", "tool_executor")
workflow.add_edge("tool_executor", "agent")
workflow.set_entry_point("agent")

# Compile the graph with increased recursion limit
app = workflow.compile()


# Function to run the agent
def run_agent(task: str) -> Dict[str, Any]:
    initial_state = GraphState(
        agent_state=AgentState(task=task, tools=tools),
        tool_invocations=[]
    )
    result = app.invoke(initial_state)
    if result.get("tool_invocations") == [END]:
        print("Task completed successfully")
    return {
        "messages": result["agent_state"].messages,
        "memory": result["agent_state"].memory.load_memory_variables({})["history"],
    }


# Example usage
task = "Book a hotel room on hotels.com for 2 days in New York City"
result = run_agent(task)

print("Agent messages:")
for message in result["messages"]:
    print(message)

print("\nAgent memory:")
print(result["memory"])

# Stop the Playwright thread
playwright_queue.put((None, None, None, None))
playwright_queue.join()
