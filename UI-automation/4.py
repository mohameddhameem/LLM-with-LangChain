import asyncio
import base64
from typing import List, Dict, Any, Optional
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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Define the state and prediction structures
class BBox(BaseModel):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str

class Prediction(BaseModel):
    action: str
    args: Optional[List[str]]

class AgentState(BaseModel):
    page: Any  # Use Any for the Page object
    input: str
    img: str
    bboxes: List[BBox]
    prediction: Optional[Prediction] = None
    scratchpad: List[SystemMessage] = Field(default_factory=list)
    observation: str = ""

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types like Page

# Define tools
async def click(state: AgentState):
    page = state.page
    click_args = state.prediction.args if state.prediction else None
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = int(click_args[0])
    try:
        bbox = state.bboxes[bbox_id]
    except IndexError:
        return f"Error: no bbox for : {bbox_id}"
    x, y = bbox.x, bbox.y
    await page.mouse.click(x, y)
    return f"Clicked {bbox_id}"

async def type_text(state: AgentState):
    page = state.page
    type_args = state.prediction.args if state.prediction else None
    if type_args is None or len(type_args) != 2:
        return f"Failed to type in element from bounding box labeled as number {type_args}"
    bbox_id = int(type_args[0])
    bbox = state.bboxes[bbox_id]
    x, y = bbox.x, bbox.y
    text_content = type_args[1]
    await page.mouse.click(x, y)
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed {text_content} and submitted"

async def take_screenshot(state: AgentState):
    page = state.page
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    await page.screenshot(path=filename)
    return f"Screenshot saved as {filename}"

# Browser annotation function
async def mark_page(page: Page):
    # You'll need to implement the JavaScript for markPage() and unmarkPage()
    await page.evaluate("markPage()")
    bboxes = await page.evaluate("getBoundingBoxes()")
    screenshot = await page.screenshot()
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }

# Agent definition
async def annotate(state: AgentState):
    marked_page = await mark_page(state.page)
    return AgentState(**{**state.dict(), **marked_page})

def format_descriptions(state: AgentState):
    labels = []
    for i, bbox in enumerate(state.bboxes):
        text = bbox.ariaLabel or bbox.text
        labels.append(f'{i} (<{bbox.type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return AgentState(**{**state.dict(), "bbox_descriptions": bbox_descriptions})

def parse(text: str) -> Prediction:
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return Prediction(action="retry", args=[f"Could not parse LLM Output: {text}"])
    action_block = text.strip().split("\n")[-1]
    action_str = action_block[len(action_prefix):]
    split_output = action_str.split(" ", 1)
    action = split_output[0].strip()
    action_input = split_output[1].strip().split(";") if len(split_output) > 1 else None
    return Prediction(action=action, args=action_input)

# LLM and agent setup
llm = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=4096)
prompt = hub.pull("wfh/web-voyager")

# Create agent using RunnablePassthrough
agent = RunnablePassthrough(assign={"prediction": format_descriptions | prompt | llm | StrOutputParser() | parse})

# Graph definition
def update_scratchpad(state: AgentState):
    old = state.scratchpad
    if old:
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(last_line.split(".")[0]) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state.observation}"
    return AgentState(**{**state.dict(), "scratchpad": [SystemMessage(content=txt)]})

graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent)
graph_builder.set_entry_point("agent")

graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Screenshot": take_screenshot,
}

for node_name, tool in tools.items():
    graph_builder.add_node(node_name, RunnableLambda(tool) | (lambda observation: {"observation": observation}))
    graph_builder.add_edge(node_name, "update_scratchpad")

def select_tool(state: AgentState):
    action = state.prediction.action if state.prediction else None
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    if action in tools:
        return action
    return "agent"  # Default to "agent" if action is not recognized

graph_builder.add_conditional_edges("agent", select_tool)
graph = graph_builder.compile()

# Main execution
async def run_automation(task: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("https://www.google.com")

        initial_state = AgentState(
            page=page,
            input=task,
            img="",
            bboxes=[],
            prediction=None,
            scratchpad=[],
            observation=""
        )
        
        async for event in graph.astream(initial_state):
            if "agent" in event:
                print(f"Action: {event['agent'].prediction.action if event['agent'].prediction else 'No prediction'}")
                print(f"Args: {event['agent'].prediction.args if event['agent'].prediction else 'No args'}")
            elif "observation" in event:
                print(f"Observation: {event['observation']}")

        await browser.close()

async def main():
    task = "Search google.com for the distance between Earth and Moon"
    await run_automation(task)

if __name__ == "__main__":
    asyncio.run(main())
