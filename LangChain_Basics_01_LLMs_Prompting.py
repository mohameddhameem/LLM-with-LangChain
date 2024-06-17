from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Initialize the OpenAI chat model with specified parameters
llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                 temperature=0.9,
                 max_tokens=256)

print("===============================================")
print("USING PROMPT TEMPLATES WITH LLM MODELS")
# Define a PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="Why do I need {topic}?"
)

# Create the prompt by filling in the template
prompt_text = prompt_template.format(topic="LLM")

# Use the invoke method to get a response from the model
response = llm.invoke(prompt_text)

# Print the response
print(response)

print("===============================================")
print("USING PROMPT TEMPLATES WITH LLM CHAIN MODELS")
### Using Prompt Templates with LLM Chain Models

restaurant_template = """
I want you to act as a naming consultant for new restaurants.

Return a list of restaurant names. Each name should be short, catchy and easy to remember. It shoud relate to the type of restaurant you are naming.

What are some good names for a restaurant that is {restaurant_desription}?
"""

prompt = PromptTemplate(
    input_variables=["restaurant_desription"],
    template=restaurant_template,
)

# An example prompt with one input variable
prompt_template = PromptTemplate(input_variables=["restaurant_desription"], template=restaurant_template)

description = "a Greek place that serves fresh lamb souvlakis and other Greek food "
description_02 = "a burger place that is themed with baseball memorabilia"
description_03 = "a cafe that has live hard rock music and memorabilia"

## Below code base is deprecated. Use the recommended code I have given below
## to see what the prompt will be like - use it only to print. no need to pass into chain
# prompt_template.format(restaurant_desription=description)
# chain = LLMChain(llm=llm,
#                  prompt=prompt_template)
# # Run the chain only specifying the input variable.
# print(chain.run(description_03))

# recommended code
chain = prompt_template | llm | StrOutputParser()
response = chain.invoke({"restaurant_desription": description_02})
print(response)

print("===============================================")
print("FEW SHORT LEARNING")
## Few Short learning
from langchain.prompts import FewShotPromptTemplate
# First, create the list of few shot examples.
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]
# Next, we specify the template to format the examples we have provided.
# We use the `PromptTemplate` class for this.
example_formatter_template = """
Word: {word}
Antonym: {antonym}\n
"""
example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)

# Finally, we create the `FewShotPromptTemplate` object.
few_shot_prompt = FewShotPromptTemplate(
    # These are the examples we want to insert into the prompt.
    examples=examples,
    # This is how we want to format the examples when we insert them into the prompt.
    example_prompt=example_prompt,
    # The prefix is some text that goes before the examples in the prompt.
    # Usually, this consists of intructions.
    prefix="Give the antonym of every input",
    # The suffix is some text that goes after the examples in the prompt.
    # Usually, this is where the user input will go
    suffix="Word: {input}\nAntonym:",
    # The input variables are the variables that the overall prompt expects.
    input_variables=["input"],
    # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
    example_separator="\n",
)

few_short_chain = few_shot_prompt | llm | StrOutputParser()
response = few_short_chain.invoke({"input": "happy"})
print(response) # sad or unhappy