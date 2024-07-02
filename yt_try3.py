import asyncio

import threading
from queue import Queue
from typing import Any, TypedDict
from typing import Dict, List

from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from playwright.sync_api import sync_playwright


# Set up your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

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


# Define Playwright operations
def navigate(page, url: str) -> str:
    page.goto(url)
    return f"Navigated to {url}"


def click(page, selector: str) -> str:
    page.click(selector)
    return f"Clicked on element with selector: {selector}"


def type_text(page, selector: str, text: str) -> str:
    page.fill(selector, text)
    return f"Typed '{text}' into element with selector: {selector}"


def extract_text(page, selector: str) -> str:
    element = page.query_selector(selector)
    if element:
        return element.inner_text()
    return "Element not found"


# Wrapper function to execute Playwright operations in the separate thread
def execute_playwright_op(func, *args, **kwargs):
    result_queue = Queue()
    playwright_queue.put((func, args, kwargs, result_queue))
    result = result_queue.get()
    if isinstance(result, Exception):
        raise result
    return result


# Define the tools
tools = [
    Tool(
        name="Navigate",
        func=lambda url: execute_playwright_op(navigate, url),
        description="Navigate to a specific URL",
    ),
    Tool(
        name="Click",
        func=lambda selector: execute_playwright_op(click, selector),
        description="Click on an element on the page",
    ),
    Tool(
        name="Type",
        func=lambda selector, text: execute_playwright_op(type_text, selector, text),
        description="Type text into an input field",
    ),
    Tool(
        name="Extract Text",
        func=lambda selector: execute_playwright_op(extract_text, selector),
        description="Extract text from an element on the page",
    ),
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
        ("system", "You are an AI assistant that helps with web automation tasks. Use the provided tools to complete the task. The available tools are: Navigate, Click, Type, and Extract Text. Respond with the tool name and input in the format: TOOL: tool_name, INPUT: tool_input. Be specific with selectors for Click and Type tools, using CSS selectors or XPath. If you need to navigate to a website, always use the Navigate tool first. If you believe the task is complete, respond with COMPLETE: reason."),
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
        
        if tool_name not in ["Navigate", "Click", "Type", "Extract Text"]:
            print(f"Invalid tool name: {tool_name}")
            agent_state.messages.append(f"Error: Invalid tool name {tool_name}")
            return {"agent_state": agent_state, "tool_invocations": []}
        
        tool_invocation = ToolInvocation(tool=tool_name, tool_input={"input": tool_input})
    else:
        print("Invalid tool invocation format")
        agent_state.messages.append("Error: Invalid tool invocation format")
        return {"agent_state": agent_state, "tool_invocations": []}
    
    agent_state.messages.append(str(result))
    
    return {
        "agent_state": agent_state,
        "tool_invocations": [tool_invocation]
    }


# Update the tool_executor function
def tool_executor(state: GraphState) -> GraphState:
    agent_state = state['agent_state']
    
    # Debug statement to print the current state of tool_invocations
    print(f"Current tool_invocations: {state['tool_invocations']}")
    
    if not state['tool_invocations']:
        # Handle the case where tool_invocations is empty
        print("No tool invocations found. Returning without executing any tool.")
        return {"agent_state": agent_state, "tool_invocations": []}
    
    tool_invocation = state['tool_invocations'][0]
    tool_executor = ToolExecutor(tools)
    
    try:
        result = tool_executor.invoke(tool_invocation)
        agent_state.memory.save_context({"human": tool_invocation.tool}, {"ai": str(result)})
        agent_state.messages.append(f"Executed {tool_invocation.tool}: {result}")
    except Exception as e:
        error_message = f"Error executing tool {tool_invocation.tool}: {str(e)}"
        print(error_message)
        agent_state.messages.append(error_message)
        
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


async def run_agent_with_timeout(task: str, timeout: int = 300) -> Dict[str, Any]:
    initial_state = GraphState(
        agent_state=AgentState(task=task, tools=tools),
        tool_invocations=[]
    )
    try:
        result = await asyncio.wait_for(app.ainvoke(initial_state, {"recursion_limit": 100}), timeout=timeout)
        return {
            "messages": result["agent_state"].messages,
            "memory": result["agent_state"].memory.load_memory_variables({})["history"],
        }
    except asyncio.TimeoutError:
        print(f"Agent execution timed out after {timeout} seconds")
        return {"messages": ["Execution timed out"], "memory": []}

# Usage
task = "Book a hotel room on hotels.com for 2 days in New York City"
result = asyncio.run(run_agent_with_timeout(task))

print("Agent messages:")
for message in result["messages"]:
    print(message)

print("\nAgent memory:")
print(result["memory"])

# Stop the Playwright thread
playwright_queue.put((None, None, None, None))
playwright_queue.join()
