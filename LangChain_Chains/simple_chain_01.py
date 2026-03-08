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

template = PromptTemplate(
    template="Generate top 5 interesting facts in short and in one sentence each about {topic}",
    input_variables=['topic']
)

topic = input('Enter any topic to get top 5 facts about it...\n')

# Knaive Approach
# prompt = template.invoke({
#     'topic': topic
# })
# result = model.invoke(prompt).content
# print('\n', parser.parse(result))

# Chains
chain = template | model | parser # LangChain Expression Language (LCEL)
result = chain.invoke({
    'topic': topic
})
print(result)

# To visualize the chain
chain.get_graph().print_ascii()