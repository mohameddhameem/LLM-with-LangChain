import json
import threading
from queue import Queue
from typing import Any, TypedDict
from typing import Dict, List
from langchain.tools import StructuredTool
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Dict, List, Any, TypedDict
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
    tools: List[StructuredTool]
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

class NavigateInput(BaseModel):
    url: str = Field(..., description="The URL to navigate to")

class ClickInput(BaseModel):
    selector: str = Field(..., description="The CSS selector of the element to click")

class TypeInput(BaseModel):
    selector: str = Field(..., description="The CSS selector of the input field")
    text: str = Field(..., description="The text to type into the input field")

class ExtractTextInput(BaseModel):
    selector: str = Field(..., description="The CSS selector of the element to extract text from")

tools = [
    StructuredTool.from_function(
        func=lambda input: execute_playwright_op(navigate, input.url),
        name="Navigate",
        description="Navigate to a specific URL",
        args_schema=NavigateInput
    ),
    StructuredTool.from_function(
        func=lambda input: execute_playwright_op(click, input.selector),
        name="Click",
        description="Click on an element on the page",
        args_schema=ClickInput
    ),
    StructuredTool.from_function(
        func=lambda input: execute_playwright_op(type_text, input.selector, input.text),
        name="Type",
        description="Type text into an input field",
        args_schema=TypeInput
    ),
    StructuredTool.from_function(
        func=lambda input: execute_playwright_op(extract_text, input.selector),
        name="Extract Text",
        description="Extract text from an element on the page",
        args_schema=ExtractTextInput
    ),
]

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
    description="Type text into an input field. Format: selector, text",
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
        ("system", "You are an AI assistant that helps with web automation tasks. Use the provided tools to complete the task. The available tools are: Navigate, Click, Type, and Extract Text. Respond with the tool name and input in JSON format. For example: TOOL: Navigate, INPUT: {\"url\": \"https://www.google.com\"}. If you believe the task is complete, respond with COMPLETE: reason."),
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
        tool_input = json.loads(input_part.strip())
        
        tool_invocation = ToolInvocation(tool=tool_name, tool_input=tool_input)
    else:
        # If the output is not in the expected format, use a default tool
        tool_invocation = ToolInvocation(tool="Extract Text", tool_input={"selector": "body"})
    
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
task = "Navigate to google.com and search for the distance between sun and earth in kilometers"
result = run_agent(task)

print("Agent messages:")
for message in result["messages"]:
    print(message)

print("\nAgent memory:")
print(result["memory"])

# Stop the Playwright thread
playwright_queue.put((None, None, None, None))
playwright_queue.join()