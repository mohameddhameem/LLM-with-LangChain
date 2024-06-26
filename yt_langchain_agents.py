# -*- coding: utf-8 -*-
"""YT LangChain - Agents.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QpvUHQzpHPvlMgBElwd5NJQ6qpryVeWE
"""
from langchain_openai import OpenAI

"""### Creating an Agent"""

from langchain.agents import load_tools
from langchain.agents import initialize_agent

llm = OpenAI(temperature=0)

"""### Loading Tools"""

tools = load_tools(["serpapi", "llm-math"], llm=llm)

# What is a tool like this
tools[1].name, tools[1].description

tools[1]

"""### Initializing the Agent"""

agent = initialize_agent(tools,
                         llm,
                         agent="zero-shot-react-description",
                         verbose=True)

agent.agent.llm_chain.prompt.template

## Standard LLM Query
agent.run("Hi How are you today?")

agent.run("Who is the United States President? What is his current age raised divided by 2?")

agent.run("What is the average age in the United States? How much less is that then the age of the current US President?")

"""## New Agent"""

tools = load_tools(["serpapi", "llm-math","wikipedia","terminal"], llm=llm)

agent = initialize_agent(tools,
                         llm,
                         agent="zero-shot-react-description",
                         verbose=True)

agent.agent.llm_chain.prompt.template

agent.run("Who is the head of DeepMind")

agent.run("What is DeepMind")

agent.run("Take the year DeepMind was founded and add 50 years. Will this year have AGI?")

agent.run("Where is DeepMind's office?")

agent.run("If I square the number for the street address of DeepMind what answer do I get?")

agent.run("What files are in my current directory?")

agent.run("Does my current directory have a file about California?")

