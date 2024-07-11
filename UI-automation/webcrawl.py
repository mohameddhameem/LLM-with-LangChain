import asyncio
import json
import base64
from typing import List, Dict
from playwright.async_api import async_playwright, Page
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit

from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.\n",      },
)
from langchain.tools.playwright.click import ClickTool
from langchain.tools.playwright.current_page import CurrentWebPageTool
from langchain.tools.playwright.extract_text import ExtractTextTool
from langchain.tools.playwright.get_elements import GetElementsTool
from langchain.tools.playwright.navigate import NavigateTool
from langchain.tools.playwright.extract_hyperlinks import ExtractHyperlinksTool
import networkx as nx
import matplotlib.pyplot as plt
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage, SystemMessage

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize GPT-4 with vision capabilities
gpt4_vision = ChatOpenAI(model="gpt-4o", max_tokens=300)

class EnhancedPlaywrightCrawlerTool(BaseTool):
    name = "EnhancedPlaywrightCrawler"
    description = "Navigates web pages, analyzes content, interacts with forms, and extracts information using Playwright and GPT-4 Vision"

    async def _arun(self, url: str) -> str:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url)
            
            # Capture screenshot
            screenshot = await page.screenshot(type='jpeg', quality=50)
            screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
            
            # Analyze page with GPT-4 Vision
            messages = [
                SystemMessage(content="You are an AI assistant specialized in analyzing web pages. Your task is to determine the type of page (e.g., login, form, content, etc.) and suggest appropriate actions."),
                HumanMessage(content=[
                    {"type": "text", "text": "What type of web page is this? What actions should be taken? If it's a form, what fields need to be filled?"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{screenshot_base64}"}
                ])
            ]
            
            response = await gpt4_vision.ainvoke(messages)
            page_analysis = response.content
            
            # Process the page based on GPT-4's analysis
            if "login" in page_analysis.lower():
                await page.fill("#username", config['username'])
                await page.fill("#password", config['password'])
                await page.click("#submit")
            elif "form" in page_analysis.lower():
                # Extract form fields from GPT-4's analysis
                form_fields = extract_form_fields(page_analysis)
                for field, selector in form_fields.items():
                    if field in config['form_values']:
                        await page.fill(selector, config['form_values'][field])
                submit_button = await page.query_selector('input[type="submit"], button[type="submit"]')
                if submit_button:
                    await submit_button.click()
                else:
                    await page.evaluate("document.querySelector('form').submit()")
            
            # Extract links
            links = await page.evaluate("""
                () => Array.from(document.querySelectorAll('a')).map(a => a.href)
            """)
            
            await browser.close()
            return json.dumps({
                "current_url": page.url,
                "links": links,
                "page_analysis": page_analysis
            })

def extract_form_fields(analysis):
    # This function should parse the GPT-4 analysis and extract form field information
    # For simplicity, we'll use a placeholder implementation
    return {
        "username": "#username",
        "password": "#password",
        "email": "#email",
        # Add more fields as needed
    }

# Create LangChain agent with enhanced tools
async def setup_browser():
    browser = await create_async_playwright_browser()
    return browser

async def main():
    browser = await setup_browser()
    
    tools = [
        EnhancedPlaywrightCrawlerTool(),
        NavigateTool(sync_browser=browser),
        ClickTool(sync_browser=browser),
        CurrentWebPageTool(sync_browser=browser),
        ExtractTextTool(sync_browser=browser),
        GetElementsTool(sync_browser=browser),
        ExtractHyperlinksTool(sync_browser=browser),
        PythonREPLTool()
    ]

    llm = ChatOpenAI(temperature=0)

    prompt = PromptTemplate.from_template(
        "You are a web crawler. Your task is to navigate through a website and extract all unique pages. "
        "Use the EnhancedPlaywrightCrawler tool to visit pages, analyze content, and extract links. "
        "Use the other browser tools to interact with the page as needed. "
        "Use the Python REPL to process data and maintain the list of visited pages and the graph. "
        "Current task: {input}"
        "Current visited pages: {visited_pages}"
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Define the crawler function
    async def crawl(state: Dict) -> Dict:
        url = state['current_url']
        result = await agent_executor.ainvoke({
            "input": f"Crawl the page: {url}",
            "visited_pages": list(state['visited_pages'])
        })
        
        page_data = json.loads(result['output'])
        current_url = page_data['current_url']
        links = page_data['links']
        page_analysis = page_data['page_analysis']
        
        state['visited_pages'].add(current_url)
        state['page_analyses'][current_url] = page_analysis
        
        for link in links:
            if link not in state['visited_pages']:
                state['graph'].add_edge(current_url, link)
        
        state['to_visit'] = list(set(links) - state['visited_pages'])
        
        return state

    # Define the main workflow
    def crawler_workflow(state: Dict) -> List[Dict]:
        if not state['to_visit']:
            return [{"type": END}]
        else:
            state['current_url'] = state['to_visit'].pop(0)
            return [{"type": "crawler", "state": state}]

    # Create the graph
    workflow = StateGraph(nodes={"crawler": crawl})

    # Define edges
    workflow.add_edge('crawler', crawler_workflow)

    # Compile the graph
    app = workflow.compile()

    # Run the crawler
    start_url = config['start_url']
    initial_state = {
        "current_url": start_url,
        "visited_pages": set(),
        "to_visit": [start_url],
        "graph": nx.DiGraph(),
        "page_analyses": {}
    }
    
    for output in app.stream(initial_state):
        if output['type'] == END:
            final_state = output['state']
            break
    
    # Output results
    print("Unique pages visited:")
    for page in final_state['visited_pages']:
        print(f"URL: {page}")
        print(f"Analysis: {final_state['page_analyses'][page]}\n")
    
    # Create and save graph visualization
    graph = final_state['graph']
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=500, node_color='lightblue', 
            font_size=8, font_weight='bold', arrows=True)
    plt.title("Web Page Flow")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('web_flow_graph.png', format='png', dpi=300, bbox_inches='tight')

    await browser.close()

asyncio.run(main())