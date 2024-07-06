import json
from typing import List, Dict, Any, TypedDict
import asyncio
from datetime import datetime
import base64
import ast

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

# Step 1: Parse the JSON file
def parse_json_log(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Tool functions (reused from the reference code)
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

# Step 2: Create base agent
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

# Step 3: Create positive test case
def format_descriptions(bboxes):
    labels = []
    for i, bbox in enumerate(bboxes):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    return "\nValid Bounding Boxes:\n" + "\n".join(labels)

def create_positive_test(steps: List[str], expected_answer: str):
    async def test_func(agent, tools):
        browser = await async_playwright().start()
        browser = await browser.chromium.launch(headless=True)
        page = await browser.new_page()
        
        state = AgentState(
            page=page,
            input="Perform the navigation steps",
            img="",
            bboxes=[],
            prediction=Prediction(action="", args=[]),
            scratchpad=[],
            observation=""
        )
        
        for step in steps:
            action, args_str = step.split(", Args: ")
            action = action.replace("Action: ", "")  # Remove "Action: " prefix
            try:
                args = ast.literal_eval(args_str)
            except:
                args = [args_str]
            state["prediction"] = {"action": action, "args": args}
            if action in tools:
                result = await tools[action](state)
                state["observation"] = result
                print(f"Step result: {result}")
            else:
                print(f"Action {action} not found in tools")
        
        # Prepare input for the agent
        agent_input = {
            "input": state["input"],
            "img": state["img"],
            "bbox_descriptions": format_descriptions(state["bboxes"]),
            "scratchpad": state["scratchpad"]
        }
        
        final_result = await agent.ainvoke(agent_input)
        assert final_result["action"] == "ANSWER", f"Expected ANSWER action, but got {final_result['action']}"
        assert final_result["args"][0] == expected_answer, f"Expected {expected_answer}, but got {final_result['args'][0]}"
        
        await browser.close()
    
    return test_func

# Step 4: Create negative test case
def create_negative_test(steps: List[str], expected_answer: str):
    async def test_func(agent, tools):
        browser = await async_playwright().start()
        browser = await browser.chromium.launch(headless=True)
        page = await browser.new_page()
        
        state = AgentState(
            page=page,
            input="Perform the navigation steps",
            img="",
            bboxes=[],
            prediction=Prediction(action="", args=[]),
            scratchpad=[],
            observation=""
        )
        
        modified_steps = steps.copy()
        for i, step in enumerate(modified_steps):
            if "Click" in step:
                action, args_str = step.split(", Args: ")
                action = action.replace("Action: ", "")  # Remove "Action: " prefix
                try:
                    args = ast.literal_eval(args_str)
                    new_args = [str(int(args[0]) + 1)]  # Click on the next element
                except:
                    new_args = [args_str]  # Fallback to treating the whole string as a single argument
                modified_steps[i] = f"{action}, Args: {new_args}"
                break
        
        for step in modified_steps:
            action, args_str = step.split(", Args: ")
            action = action.replace("Action: ", "")  # Remove "Action: " prefix
            try:
                args = ast.literal_eval(args_str)
            except:
                args = [args_str]
            state["prediction"] = {"action": action, "args": args}
            if action in tools:
                result = await tools[action](state)
                state["observation"] = result
                print(f"Step result: {result}")
            else:
                print(f"Action {action} not found in tools")
        
        # Prepare input for the agent
        agent_input = {
            "input": state["input"],
            "img": state["img"],
            "bbox_descriptions": format_descriptions(state["bboxes"]),
            "scratchpad": state["scratchpad"]
        }
        
        final_result = await agent.ainvoke(agent_input)
        assert final_result["action"] != "ANSWER" or final_result["args"][0] != expected_answer, f"Expected a different answer, but got {final_result['args'][0]}"
        
        await browser.close()
    
    return test_func

# Step 5: Implement test runner
async def run_tests(test_cases):
    agent, tools = create_base_agent()
    for i, test in enumerate(test_cases):
        print(f"Running test case {i+1}")
        try:
            await test(agent, tools)
            print("Test passed")
        except AssertionError as e:
            print(f"Test failed: {str(e)}")
        except Exception as e:
            print(f"An error occurred during the test: {str(e)}")
        print()

if __name__ == "__main__":
    log_data = parse_json_log("navigation_log.json")
    steps = [step for step in log_data["steps"] if not step.startswith("Observation:")]
    expected_answer = log_data["final_answer"]
    
    positive_test = create_positive_test(steps, expected_answer)
    negative_test = create_negative_test(steps, expected_answer)
    
    asyncio.run(run_tests([positive_test,negative_test]))