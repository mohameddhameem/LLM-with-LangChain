import json
from typing import List, Dict, Any, TypedDict
import asyncio
from datetime import datetime
import base64

from langchain_core.messages import SystemMessage
from playwright.async_api import Page, async_playwright
import nest_asyncio

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph

nest_asyncio.apply()

# Type definitions
class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str

class Prediction(TypedDict):
    action: str
    args: List[str]

class AgentState(TypedDict):
    page: Page
    input: str
    img: str
    bboxes: List[BBox]
    prediction: Prediction
    scratchpad: List[SystemMessage]
    observation: str

# Tool functions
async def click(state: AgentState):
    page = state["page"]
    click_args = state["prediction"]["args"]
    if len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = int(click_args[0])
    try:
        bbox = state["bboxes"][bbox_id]
        element_type = bbox.get("type", "unknown")
        element_text = bbox.get("text", "")[:30]  # Truncate long text
        x, y = bbox["x"], bbox["y"]
        await page.mouse.click(x, y)
        return f"Clicked element {bbox_id} on {page.url}: <{element_type}> with text '{element_text}'"
    except Exception as e:
        return f"Error clicking bbox {bbox_id} on {page.url}: {str(e)}"

async def type_text(state: AgentState):
    page = state["page"]
    type_args = state["prediction"]["args"]
    if len(type_args) != 2:
        return f"Failed to type in element from bounding box labeled as number {type_args}"
    bbox_id = int(type_args[0])
    bbox = state["bboxes"][bbox_id]
    element_type = bbox.get("type", "unknown")
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    await page.mouse.click(x, y)
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed '{text_content}' into element {bbox_id} on {page.url}: <{element_type}> and submitted"

# Create base agent
def create_base_agent():
    llm = ChatOpenAI(model="gpt-4", max_tokens=4096)
    
    tools = {
        "Click": click,
        "Type": type_text,
        # Add more tools as needed
    }
    
    from langchain import hub
    prompt = hub.pull("wfh/web-voyager")
    
    agent = prompt | llm | StrOutputParser() | parse
    
    return agent, tools

# Helper function to parse LLM output
def parse(text: str) -> dict:
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]

    action_str = action_block[len(action_prefix):]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [inp.strip().strip("[]") for inp in action_input.strip().split(";")]
    return {"action": action, "args": action_input}

# Define tools dictionary
tools = {
    "Click": click,
    "Type": type_text,
    # Add more tools as needed
}

# Define state graph
state_graph = StateGraph(
    start_state="start",
    end_state=END,
    states={
        "start": {
            "prompt": "What would you like to do?",
            "tools": tools,
            "next_state": "navigate"
        },
        "navigate": {
            "prompt": "Please provide the navigation steps or query.",
            "tools": tools,
            "next_state": "start"
        }
    }
)

# Run the agent
async def run_agent():
    browser = await async_playwright().start()
    browser = await browser.chromium.launch(headless=False)
    page = await browser.new_page()

    initial_state = AgentState(
        page=page,
        input="",
        img="",
        bboxes=[],
        prediction=Prediction(action="", args=[]),
        scratchpad=[],
        observation=""
    )

    agent, _ = create_base_agent()
    await state_graph.run(agent, initial_state)

    await browser.close()

if __name__ == "__main__":
    asyncio.run(run_agent())