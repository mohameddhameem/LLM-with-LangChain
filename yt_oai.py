import threading
from queue import Queue
from typing import Dict, List, Any
from typing_extensions import TypedDict
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from playwright.sync_api import sync_playwright
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# Set up your OpenAI API key
# import os
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Create the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that helps with web automation tasks. Use the provided tools to complete the task."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    MessagesPlaceholder(variable_name="chat_history"),
])


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

from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field

# Define input schemas for structured tools
class NavigateInput(BaseModel):
    url: str = Field(..., description="The URL to navigate to")

class ClickInput(BaseModel):
    selector: str = Field(..., description="The CSS selector of the element to click")

class TypeInput(BaseModel):
    selector: str = Field(..., description="The CSS selector of the input field")
    text: str = Field(..., description="The text to type into the input field")

class ExtractTextInput(BaseModel):
    selector: str = Field(..., description="The CSS selector of the element to extract text from")

# Define the tools
class NavigateTool(BaseTool):
    name = "Navigate"
    description = "Navigate to a specific URL"
    args_schema = NavigateInput

    def _run(self, url: str):
        return execute_playwright_op(navigate, url)

class ClickTool(BaseTool):
    name = "Click"
    description = "Click on an element on the page"
    args_schema = ClickInput

    def _run(self, selector: str):
        return execute_playwright_op(click, selector)

class TypeTool(BaseTool):
    name = "Type"
    description = "Type text into an input field"
    args_schema = TypeInput

    def _run(self, selector: str, text: str):
        return execute_playwright_op(type_text, selector, text)

class ExtractTextTool(BaseTool):
    name = "Extract Text"
    description = "Extract text from an element on the page"
    args_schema = ExtractTextInput

    def _run(self, selector: str):
        return execute_playwright_op(extract_text, selector)

# Create instances of the tools
tools = [
    NavigateTool(),
    ClickTool(),
    TypeTool(),
    ExtractTextTool()
]

# Create the agent
system_message = SystemMessage(
    content="You are an AI assistant that helps with web automation tasks. Use the provided tools to complete the task."
)

prompt = [
    system_message,
    HumanMessage(content="Complete the following task: {input}\nCurrent conversation:\n{chat_history}")
]

agent = OpenAIFunctionsAgent(
    llm=ChatOpenAI(temperature=0),
    tools=tools,
    prompt=prompt
)

# Create an AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Function to run the agent
def run_agent(task: str) -> Dict[str, Any]:
    memory = ConversationBufferMemory(return_messages=True)
    result = agent_executor.invoke({
        "input": task,
        "chat_history": memory.chat_memory.messages
    })
    return result

# Example usage
task = "Navigate to google.com and search for the distance between sun and earth in kilometers"
result = run_agent(task)

print("\nAgent result:")
print(result)

# Stop the Playwright thread
playwright_queue.put((None, None, None, None))
playwright_queue.join()