import os
from typing import List, Tuple, Any, Dict, TypedDict
from typing import Dict, Tuple, List
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain.agents import AgentExecutor, Tool
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain.memory import ConversationBufferMemory
from playwright.sync_api import Page
from playwright.sync_api import sync_playwright

# Create a synchronous Playwright instance and browser
playwright = sync_playwright().start()
browser = playwright.chromium.launch(headless=True)
page = browser.new_page()

# Set up your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"


# Define custom Playwright tools
def navigate(url: str) -> str:
    page.goto(url)
    return f"Navigated to {url}"

def click(selector: str) -> str:
    page.click(selector)
    return f"Clicked on element with selector: {selector}"

def type(selector: str, text: str) -> str:
    page.fill(selector, text)
    return f"Typed '{text}' into element with selector: {selector}"

def extract_text(selector: str) -> str:
    element = page.query_selector(selector)
    if element:
        return element.inner_text()
    return "Element not found"

# Define the tools
tools = [
    Tool(
        name="Navigate",
        func=navigate,
        description="Navigate to a specific URL",
    ),
    Tool(
        name="Click",
        func=click,
        description="Click on an element on the page",
    ),
    Tool(
        name="Type",
        func=type,
        description="Type text into an input field",
    ),
    Tool(
        name="Extract Text",
        func=extract_text,
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
        ("system", "You are an AI assistant that helps with web automation tasks. Use the provided tools to complete the task. The available tools are: Navigate, Click, Type, and Extract Text. Respond with the tool name and input in the format: TOOL: tool_name, INPUT: tool_input"),
        ("human", "{task}"),
        ("human", "Current conversation:\n{memory}\n\nHuman: What should I do next?"),
    ])
    chain = prompt | llm
    result = chain.invoke({
        "task": agent_state.task,
        "memory": agent_state.memory.load_memory_variables({})["history"],
    })
    
    # Parse the LLM output to create a ToolInvocation
    content = result.content
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
    result = tool_executor.invoke(tool_invocation)
    agent_state.memory.save_context({"human": tool_invocation.tool}, {"ai": str(result)})
    agent_state.messages.append(f"Executed {tool_invocation.tool}: {result}")
    return {"agent_state": agent_state, "tool_invocations": []}

# Create the graph
workflow = StateGraph(GraphState)

# Add agent and tool_executor nodes
workflow.add_node("agent", agent)
workflow.add_node("tool_executor", tool_executor)

# Add edges
workflow.add_edge("agent", "tool_executor")
workflow.add_edge("tool_executor", "agent")

# Set the entry point
workflow.set_entry_point("agent")

# Compile the graph
app = workflow.compile()

# Function to run the agent
def run_agent(task: str) -> Dict[str, Any]:
    initial_state = GraphState(
        agent_state=AgentState(task=task, tools=tools),
        tool_invocations=[]
    )
    result = app.invoke(initial_state)
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

# At the end of your script, after you're done with the browser:
page.close()
browser.close()
playwright.stop()