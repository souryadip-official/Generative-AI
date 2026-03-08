# Objective: Find out the best facts from a list of facts
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-3-27b-it",
    huggingfacehub_api_token = huggingface_api_key,
)
model = ChatHuggingFace(llm = llm)
parser = StrOutputParser()

template1 = PromptTemplate(
    template="Generate top 10 interesting facts in short and in one sentence each about {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Figure out and select the best 2 facts from this facts and keep them to the point and one liners: {facts}",
    input_variables=['facts']
)

topic = input('Enter any topic: ')

# Chains
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({
    'topic': topic
})
print(result)

# To visualize the chain
chain.get_graph().print_ascii()

# Simple chain is also called as a sequential chain as output of one stage becomes the input for the next and this entire flow happens stage by stage.