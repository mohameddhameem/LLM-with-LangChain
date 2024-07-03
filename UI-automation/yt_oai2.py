import threading
from queue import Queue
from typing import Dict, List, Any

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from playwright.sync_api import sync_playwright

# Set up your OpenAI API key
# import os
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

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

# Define input schemas for structured tools
class NavigateInput(BaseModel):
    url: str = Field(..., description="The URL to navigate to")

class ClickInput(BaseModel):
    selector: str = Field(..., description="The CSS selector of the element to click")

class TypeInput(BaseModel):
    selector_and_text: str = Field(..., description="The CSS selector of the input field and the text to type, separated by a comma")

class ExtractTextInput(BaseModel):
    selector: str = Field(..., description="The CSS selector of the element to extract text from")


# Define the tools
class NavigateTool(BaseTool):
    name = "navigate"
    description = "Navigate to a specific URL"
    args_schema = NavigateInput

    def _run(self, url: str):
        return execute_playwright_op(navigate, url)

class ClickTool(BaseTool):
    name = "click"
    description = "Click on an element on the page"
    args_schema = ClickInput

    def _run(self, selector: str):
        return execute_playwright_op(click, selector)

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

class TypeTool(BaseTool):
    name = "type"
    description = "Type text into an input field. Provide the selector and text separated by a comma."
    args_schema = TypeInput
    playwright_timeout: float = 30000  # 30 seconds timeout

    def _run(self, selector_and_text: str):
        try:
            selector, text = selector_and_text.split(',', 1)
            selector = selector.strip()
            text = text.strip()
            
            def type_with_retry(page, selector: str, text: str) -> str:
                try:
                    page.fill(selector, text, timeout=self.playwright_timeout)
                    return f"Typed '{text}' into element with selector: {selector}"
                except PlaywrightTimeoutError:
                    return f"Unable to type into element '{selector}'. Element not found or not interactable."

            return execute_playwright_op(type_with_retry, selector, text)
        except ValueError:
            return "Error: Please provide both the selector and text, separated by a comma."

class ExtractTextTool(BaseTool):
    name = "extract_text"
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

# Create the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that helps with web automation tasks. Use the provided tools to complete the task. The available tools are: navigate, click, type, and extract_text. For the 'type' tool, provide the selector and text separated by a comma."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    MessagesPlaceholder(variable_name="chat_history"),
])

# Create the memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the agent
llm = ChatOpenAI(temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)

# Create an AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Function to run the agent
def run_agent(task: str) -> Dict[str, Any]:
    result = agent_executor.invoke({
        "input": task,
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