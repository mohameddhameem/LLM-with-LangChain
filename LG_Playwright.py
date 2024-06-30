from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_openai import ChatOpenAI
async_browser = create_async_playwright_browser(headless=False)
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = toolkit.get_tools()
for tool in tools:
    print(tool.name)

from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import Ollama
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent

prompt = hub.pull("hwchase17/structured-chat-agent")
# save the prompt
prompt.save("structured_chat_agent_prompt.json")


# llm = ChatOpenAI(model_name='gpt-3.5-turbo',
#                  temperature=0.9,
#                  max_tokens=256)

# agent = create_structured_chat_agent(llm, tools, prompt)
# agent_chain = AgentExecutor(agent=agent, tools=tools, verbose=True,
#                             handle_parsing_errors=True,
#                             return_intermediate_steps=True)

# async def main():
#     response = await agent_chain.ainvoke({"input": "What are the headers on python.langchain.com and list all hyperlinks?"})
#     print(response['output'])


# import asyncio

# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())