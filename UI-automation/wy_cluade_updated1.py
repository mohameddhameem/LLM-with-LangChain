import os
import asyncio
import platform
import base64
from getpass import getpass
from typing import List, Optional, TypedDict
import json
from datetime import datetime

from langchain_core.messages import BaseMessage, SystemMessage
from playwright.async_api import Page, async_playwright
import nest_asyncio

from langchain_core.runnables import chain as chain_decorator
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph
import re
from IPython import display

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
    args: Optional[List[str]]

class AgentState(TypedDict):
    page: Page
    input: str
    img: str
    bboxes: List[BBox]
    prediction: Prediction
    scratchpad: List[BaseMessage]
    observation: str



def save_navigation_info(question, steps, final_answer, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"navigation_log_{timestamp}.json"
    
    log_data = {
        "question": question,
        "steps": steps,
        "final_answer": final_answer,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    print(f"Navigation information saved to {filename}")

# Tool functions
async def click(state: AgentState):
    page = state["page"]
    click_args = state["prediction"]["args"]
    if click_args is None or len(click_args) != 1:
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
    if type_args is None or len(type_args) != 2:
        return f"Failed to type in element from bounding box labeled as number {type_args}"
    bbox_id = int(type_args[0])
    bbox = state["bboxes"][bbox_id]
    element_type = bbox.get("type", "unknown")
    x, y = bbox["x"], bbox["y"]
    text_content = type_args[1]
    await page.mouse.click(x, y)
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    return f"Typed '{text_content}' into element {bbox_id} on {page.url}: <{element_type}> and submitted"

async def scroll(state: AgentState):
    page = state["page"]
    scroll_args = state["prediction"]["args"]
    if scroll_args is None or len(scroll_args) != 2:
        return "Failed to scroll due to incorrect arguments."

    target, direction = scroll_args

    if target.upper() == "WINDOW":
        scroll_amount = 500
        scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
        return f"Scrolled {direction} in window on {page.url}"
    else:
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        element_type = bbox.get("type", "unknown")
        x, y = bbox["x"], bbox["y"]
        scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)
        return f"Scrolled {direction} in element {target_id} on {page.url}: <{element_type}>"

async def call_agent(question: str, page, max_steps: int = 150):
    event_stream = graph.astream(
        {
            "page": page,
            "input": question,
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps,
        },
    )
    final_answer = None
    steps = []
    try:
        async for event in event_stream:
            if "agent" not in event:
                continue
            pred = event["agent"].get("prediction") or {}
            action = pred.get("action")
            action_input = pred.get("args")
            
            # Get the current URL
            current_url = await page.evaluate("window.location.href")
            
            # Create a more detailed step description
            step_description = f"Action: {action}, Args: {action_input}, URL: {current_url}"
            steps.append(step_description)
            
            print("\nCurrent state:")
            print("\n".join(steps))
            print(f"Latest action: {step_description}")
            
            # Display the image
            display.display(display.Image(base64.b64decode(event["agent"]["img"])))
            
            if "ANSWER" in action:
                final_answer = action_input[0]
                break
            
            # If there's an observation, add it to the steps
            if "observation" in event:
                observation = event['observation']
                steps.append(f"Observation: {observation}")
                print(f"Observation: {observation}")
    finally:
        await event_stream.aclose()
    
    # Save the navigation information
    save_navigation_info(question, steps, final_answer)
    
    return final_answer

async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."

async def go_back(state: AgentState):
    page = state["page"]
    await page.go_back()
    return f"Navigated back a page to {page.url}."

async def to_google(state: AgentState):
    page = state["page"]
    await page.goto("https://www.google.com/")
    return "Navigated to google.com."

# JavaScript for page annotation
with open("mark_page.js") as f:
    mark_page_script = f.read()

@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            await asyncio.sleep(3)
    screenshot = await page.screenshot()
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }

async def annotate(state):
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}

def format_descriptions(state):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}

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

# LLM and agent setup
prompt = hub.pull("wfh/web-voyager")
llm = ChatOpenAI(model="gpt-4o", max_tokens=4096)
agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | prompt | llm | StrOutputParser() | parse
)

def update_scratchpad(state: AgentState):
    old = state.get("scratchpad")
    if old:
        txt = old[0].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"
    return {**state, "scratchpad": [SystemMessage(content=txt)]}

# Graph setup
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent)
graph_builder.set_entry_point("agent")
graph_builder.add_node("update_scratchpad", update_scratchpad)
graph_builder.add_edge("update_scratchpad", "agent")

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
}

for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        RunnableLambda(tool) | (lambda observation: {"observation": observation}),
    )
    graph_builder.add_edge(node_name, "update_scratchpad")

def select_tool(state: AgentState):
    action = state["prediction"]["action"]
    if action == "ANSWER":
        return END
    if action == "retry":
        return "agent"
    return action

graph_builder.add_conditional_edges("agent", select_tool)
graph = graph_builder.compile()

# Main execution
async def main():
    browser = await async_playwright().start()
    browser = await browser.chromium.launch(headless=False, args=None)
    page = await browser.new_page()
    await page.goto("https://www.google.com")

    questions = [
        #"Could you explain the WebVoyager paper (on arxiv)?",
        #"Please explain the today's XKCD comic for me. Why is it funny?",
        "tell me a joke"
        #"What are the latest blog posts from langchain?",
        #"Could you check google maps to see when i should leave to get to SFO by 7 o'clock? starting from SF downtown.",
    ]

    try:
        for i, question in enumerate(questions):
            print(f"\nProcessing question {i+1}: {question}")
            try:
                res = await call_agent(question, page)
                print(f"Final response: {res}")
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
    finally:
        await browser.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")