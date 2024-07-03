import asyncio
from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain import hub
from langchain.schema import SystemMessage
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from datetime import datetime
from playwright.async_api import Page, async_playwright

# Global variables to store browser context and page
browser_context = None
current_page = None

async def initialize_browser():
    global browser_context, current_page
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    browser_context = await browser.new_context()
    current_page = await browser_context.new_page()
    return current_page

async def get_current_page():
    global current_page
    if current_page is None:
        current_page = await initialize_browser()
    return current_page

class ScreenshotTool(BaseTool):
    name = "take_screenshot"
    description = "Take a screenshot of the current webpage"

    def _run(self, page: Page) -> str:
        return asyncio.get_event_loop().run_until_complete(self._arun(page))

    async def _arun(self, page: Page) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        await page.screenshot(path=filename)
        return f"Screenshot saved as {filename}"

class AgentState(BaseModel):
    input: str
    agent_outcome: Dict[str, Any] = Field(default_factory=dict)
    attempts: int = 0
    max_attempts: int = 3
    navigation_path: List[str] = Field(default_factory=list)

def agent_node(state: AgentState):
    if state.attempts >= state.max_attempts:
        return END

    async def async_operations():
        page = await get_current_page()
        
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=page.context.browser)
        tools = toolkit.get_tools()
        tools.append(ScreenshotTool())

        llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7)
        prompt = hub.pull("hwchase17/structured-chat-agent")
        agent = create_structured_chat_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        result = await agent_executor.ainvoke({
            "input": state.input,
            "chat_history": [SystemMessage(content="You are an AI assistant tasked with web automation. Always take a screenshot after each action.")],
        })

        screenshot_tool = next(tool for tool in tools if tool.name == "take_screenshot")
        await screenshot_tool._arun(page)

        return result

    result = asyncio.get_event_loop().run_until_complete(async_operations())

    state.agent_outcome = result
    state.attempts += 1
    state.navigation_path.append(result['output'])

    if "I couldn't find the information" in result['output'] or "error" in result['output'].lower():
        return "retry"
    else:
        return "success"

def retry_node(state: AgentState):
    state.input = f"Retry the task: {state.input}. Learn from previous attempt and correct any mistakes."
    return "agent"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("retry", retry_node)
workflow.add_edge("agent", "retry")
workflow.add_edge("retry", "agent")
workflow.set_entry_point("agent")
graph = workflow.compile()

async def run_automation(task: str):
    await initialize_browser()
    
    initial_state = AgentState(input=task)
    result = graph.invoke(initial_state)
    # result = graph.run(initial_state)
    
    print("Task completed!")
    print("Navigation path:")
    for step in result.navigation_path:
        print(f"- {step}")

    with open("navigation_path.txt", "w") as f:
        f.write("\n".join(result.navigation_path))

    print("Navigation path saved to navigation_path.txt")

    global browser_context
    if browser_context:
        await browser_context.close()

async def main():
    task = "Search google.com for the distance between Earth and Moon"
    await run_automation(task)

if __name__ == "__main__":
    asyncio.run(main())