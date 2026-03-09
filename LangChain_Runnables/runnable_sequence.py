import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)
parser = StrOutputParser()

template1 = PromptTemplate(
    template='Write a joke on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="""
    Explain the joke in simple words.
    The format of the explanation should be:
    1. Original Joke (RAW)
    2. Explanation of the joke
    Joke is \"{joke}\"""",
    input_variables=['joke']
)

topic = input('Enter any topic: ')
chain = RunnableSequence(template1, model, parser, template2, model, parser)
result = chain.invoke({'topic': topic})
print(result)

chain.get_graph().draw_png('runnable_sequence_chain.png')