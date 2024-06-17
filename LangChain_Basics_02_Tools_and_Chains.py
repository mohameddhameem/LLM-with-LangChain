from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough

# Initialize the OpenAI chat model with specified parameters
llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                 temperature=0.9,
                 max_tokens=256)

print("===============================================")
print("Basic LLMChain - Fact Extraction")
print("===============================================")

article = '''Coinbase, the second-largest crypto exchange by trading volume, released its Q4 2022 earnings on Tuesday, giving shareholders and market players alike an updated look into its financials. In response to the report, the company's shares are down modestly in early after-hours trading.In the fourth quarter of 2022, Coinbase generated $605 million in total revenue, down sharply from $2.49 billion in the year-ago quarter. Coinbase's top line was not enough to cover its expenses: The company lost $557 million in the three-month period on a GAAP basis (net income) worth -$2.46 per share, and an adjusted EBITDA deficit of $124 million.Wall Street expected Coinbase to report $581.2 million in revenue and earnings per share of -$2.44 with adjusted EBITDA of -$201.8 million driven by 8.4 million monthly transaction users (MTUs), according to data provided by Yahoo Finance.Before its Q4 earnings were released, Coinbase's stock had risen 86% year-to-date. Even with that rally, the value of Coinbase when measured on a per-share basis is still down significantly from its 52-week high of $206.79.That Coinbase beat revenue expectations is notable in that it came with declines in trading volume; Coinbase historically generated the bulk of its revenues from trading fees, making Q4 2022 notable. Consumer trading volumes fell from $26 billion in the third quarter of last year to $20 billion in Q4, while institutional volumes across the same timeframe fell from $133 billion to $125 billion.The overall crypto market capitalization fell about 64%, or $1.5 trillion during 2022, which resulted in Coinbase's total trading volumes and transaction revenues to fall 50% and 66% year-over-year, respectively, the company reported.As you would expect with declines in trading volume, trading revenue at Coinbase fell in Q4 compared to the third quarter of last year, dipping from $365.9 million to $322.1 million. (TechCrunch is comparing Coinbase's Q4 2022 results to Q3 2022 instead of Q4 2021, as the latter comparison would be less useful given how much the crypto market has changed in the last year; we're all aware that overall crypto activity has fallen from the final months of 2021.)There were bits of good news in the Coinbase report. While Coinbase's trading revenues were less than exuberant, the company's other revenues posted gains. What Coinbase calls its "subscription and services revenue" rose from $210.5 million in Q3 2022 to $282.8 million in Q4 of the same year, a gain of just over 34% in a single quarter.And even as the crypto industry faced a number of catastrophic events, including the Terra/LUNA and FTX collapses to name a few, there was still growth in other areas. The monthly active developers in crypto have more than doubled since 2020 to over 20,000, while major brands like Starbucks, Nike and Adidas have dived into the space alongside social media platforms like Instagram and Reddit.With big players getting into crypto, industry players are hoping this move results in greater adoption both for product use cases and trading volumes. Although there was a lot of movement from traditional retail markets and Web 2.0 businesses, trading volume for both consumer and institutional users fell quarter-over-quarter for Coinbase.Looking forward, it'll be interesting to see if these pieces pick back up and trading interest reemerges in 2023, or if platforms like Coinbase will have to keep looking elsewhere for revenue (like its subscription service) if users continue to shy away from the market.
'''

fact_extraction_prompt = PromptTemplate(
    input_variables=["text_input"],
    template="Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep them short sentences. :\n\n {text_input}"
)

# recommended code
fact_extraction_chain = fact_extraction_prompt | llm | StrOutputParser()
# facts = fact_extraction_chain.invoke(article)
# print(facts)

print("===============================================")
print("Rewrite as a summary from the facts")
print("===============================================")
investor_update_prompt = PromptTemplate(
    input_variables=["facts"],
    template="You are a Goldman Sachs analyst. Take the following list of facts and use them to write a short paragrah for investors. Don't leave out key info:\n\n {facts}"
)

investor_update_chain = investor_update_prompt | llm | StrOutputParser()
# investor_update = investor_update_chain.invoke(facts)
# print(investor_update)

print("===============================================")
print("Triplets Extraction")
print("===============================================")

triples_prompt = PromptTemplate(
    input_variables=["facts"],
    template="Take the following list of facts and turn them into triples for a knowledge graph:\n\n {facts}"
)

triples_chain = triples_prompt | llm | StrOutputParser()
# triples = triples_chain.invoke(facts)
# print(triples)


print("===============================================")
print("Chain all together")
print("===============================================")

# Learning - as per https://github.com/langchain-ai/langchain/issues/21962 we can simply chain without SequentialChain
# full_chain = SimpleSequentialChain(chains=[fact_extraction_chain, investor_update_chain], verbose=True)
# final_reponse = full_chain.run(article)

# create a SimpleSequentialChain with fact_extraction_chain and investor_update_chain
# full_chain = SequentialChain(chains=[fact_extraction_chain, investor_update_chain, triples_chain], verbose=True)
# final_response = full_chain.invoke()

# Recommended code - v1
full_chain = fact_extraction_chain | investor_update_chain | StrOutputParser()
final_response = full_chain.invoke({"text_input": article})
print(final_response)

# Recommended code - v2 - Seems very good for prod implementation
# Refer - https://github.com/langchain-ai/langchain/discussions/20137

chain = (
        {"facts": fact_extraction_chain}
        | RunnablePassthrough.assign(review=investor_update_chain))

response = chain.invoke({"text_input": article})
print(response)
