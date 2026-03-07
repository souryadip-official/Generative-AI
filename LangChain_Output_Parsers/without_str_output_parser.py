import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token = huggingface_api_key,
)

model = ChatHuggingFace(llm = llm)
# Say our job is to ask the user for any topic, and ask the LLM to generate a detailed report on it. Then call the LLM again on the detailed report asking it to summarize it in 4-5 lines and then display it to the user

template1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Summarize this entire report \"{report}\" in four to five lines.",
    input_variables=['report']
)

topic = input('Enter any topic: ')
prompt1 = template1.invoke({
    'topic': topic
})
report = model.invoke(prompt1).content

prompt2 = template2.invoke({
    'report': report
})
summary = model.invoke(prompt2).content
print(summary)