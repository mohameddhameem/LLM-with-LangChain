from langchain.chains.conversation.memory import ConversationBufferMemory
# from langchain import OpenAI
from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain_core.prompts import PromptTemplate

llm = OpenAI(model_name='gpt-3.5-turbo-instruct',
             temperature=0,
             max_tokens=256)

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory
)

conversation.predict(input="Hi there! I am Dhameem")

conversation.predict(input="How are you today?")

conversation.predict(input="I'm good thank you. Can you help me with some customer support?")

print(conversation.memory.buffer)

# ## Basic Conversation with ConversationSummaryMemory


from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(model_name='gpt-3.5-turbo-instruct',
             temperature=0,
             max_tokens=256)

summary_memory = ConversationSummaryMemory(llm=OpenAI())

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=summary_memory
)

conversation.predict(input="Hi there! I am Dhameem")

conversation.predict(input="How are you today?")

conversation.predict(input="Can you help me with some customer support?")

print(conversation.memory.buffer)

# ## Basic Conversation with ConversationBufferWindowMemory


from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(model_name='gpt-3.5-turbo-instruct',
             temperature=0,
             max_tokens=256)

# We set a low k=2, to only keep the last 2 interactions in memory
window_memory = ConversationBufferWindowMemory(k=2)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=window_memory
)

conversation.predict(input="Hi there! I am Dhameem")

conversation.predict(input="I am looking for some customer support")

conversation.predict(input="My TV is not working.")

conversation.predict(input="When I turn it on it makes some weird sounds and then goes black")

print(conversation.memory.buffer)

# ## Basic Conversation with ConversationSummaryBufferMemory
# 
# "It keeps a buffer of recent interactions in memory, but rather than just completely flushing old interactions it compiles them into a summary and uses both"


from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(model_name='gpt-3.5-turbo-instruct',
             temperature=0,
             max_tokens=512)

# Setting k=2, will only keep the last 2 interactions in memory
# max_token_limit=40 - token limits needs transformers installed
memory = ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40)

conversation_with_summary = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation_with_summary.predict(input="Hi there! I am Dhameem")

conversation_with_summary.predict(input="I need help with my broken TV")

conversation_with_summary.predict(input="It makes weird sounds when i turn it on and then goes black")

conversation_with_summary.predict(input="It seems to be Hardware")

print(conversation_with_summary.memory.moving_summary_buffer)

# ## Basic Conversation with Conversation Knowledge Graph Memory


from langchain.chains.conversation.memory import ConversationKGMemory
from langchain import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(model_name='gpt-3.5-turbo-instruct',
             temperature=0,
             max_tokens=256)

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context.
If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
)

conversation_with_kg = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=prompt,
    memory=ConversationKGMemory(llm=llm)
)

conversation_with_kg.predict(input="Hi there! I am Dhameem")

conversation_with_kg.predict(input="My TV is broken and I need some customer assistance")

conversation_with_kg.predict(input="It makes weird sounds when i turn it on and then goes black")

conversation_with_kg.predict(input="Yes it is and it is still under warranty. my warranty number is A512423")

# NetworkX Entity graph
import networkx as nx
import matplotlib.pyplot as plt

# nx.draw(conversation_with_kg.memory.kg, with_labels=True)
# plt.show()

print(conversation_with_kg.memory.kg)
print(conversation_with_kg.memory.kg.get_triples())

# ## Entity Memory


from langchain import OpenAI, ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from pydantic import BaseModel
from typing import List, Dict, Any

## The propmpt
ENTITY_MEMORY_CONVERSATION_TEMPLATE.template

llm = OpenAI(model_name='gpt-3.5-turbo-instruct',
             temperature=0,
             max_tokens=256)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm)
)

conversation.predict(input="Hi I am Dhameem. My TV is broken but it is under warranty.")

conversation.predict(input="How can I get it fixed. The warranty number is A512453")

conversation.memory.entity_cache

conversation.predict(input="Can you send the repair person call Dave to fix it?.")

conversation.memory.entity_cache

from pprint import pprint

pprint(conversation.memory.store)
