# Objective: Create a parallel chain of generating notes and quiz on a topic
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel # Helps us execute multiple chains parallely
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm1 = HuggingFaceEndpoint(
    repo_id = "google/gemma-3-27b-it",
    huggingfacehub_api_token = huggingface_api_key,
)
model1 = ChatHuggingFace(llm = llm1)

llm2 = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token = huggingface_api_key,
)
model2 = ChatHuggingFace(llm = llm2)

template1 = PromptTemplate(
    template="""
    Generate a clear explanation of {topic} in **under 500 words**.
    Include:
    1. Idea behind
    2. Mathematical intuition
    3. Key formulas
    4. Applications""",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Below is a detailed information regarding a topic. Please generate a short and simple summary notes consising of one line sentences to summarize this entire idea and include the key points and points to remember and everything wrapped under 500 words.\n{information}",
    input_variables=['information']
)

template3 = PromptTemplate(
    template="Below is a detailed information regarding a topic. Generate a 5 short questions only. No need to provide the answer so as to test the entire concept clarity of the information.\n{information}",
    input_variables=['information']
)

template4 = PromptTemplate(
    template="""
    You are creating a study document.
    Include BOTH sections.
    SECTION 1: Summary Notes (Do not rewrite or expand the content.)
    {summary_notes}
    SECTION 2: Short Questions (Remember no need to include the answer key)
    {question}""",
    input_variables=['summary_notes', 'question']
)
parser = StrOutputParser()
topic = input('Enter any topic you want to learn: ')

# Chains
detailed_info_chain = template1 | model1 | parser
parallel_chain = RunnableParallel({
    'summary_notes': template2 | model1 | parser,
    'question': template3 | model2 | parser
})
merging_chain = template4 | model1 | parser

final_chain = detailed_info_chain | parallel_chain | merging_chain
result = final_chain.invoke({'topic': topic})
print(result)

# To visualize the chain
print(final_chain.get_graph().draw_ascii())