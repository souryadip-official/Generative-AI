import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
    temperature=0 # for lower randomness
)
strparser = StrOutputParser()

loader = TextLoader(file_path='cricket.txt', encoding='utf-8')
documents = loader.load() # python list of documents
document = documents[0] # <class 'langchain_core.documents.base.Document'>
content = document.page_content

template = PromptTemplate(
    template='Summarize the following content in 100 words\nContent =  "{content}"',
    input_variables=['content'],
)
chain = template | model | strparser
result = chain.invoke({'content': content})
print(result)