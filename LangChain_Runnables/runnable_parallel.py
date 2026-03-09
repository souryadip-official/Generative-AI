import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)
parser = StrOutputParser()

template1 = PromptTemplate(
    template='Give me a summary of personal background of the cricketer {cricketer} in 100 words',
    input_variables=['cricketer']
)

template2 = PromptTemplate(
    template='Give me a summary of professional background of the cricketer {cricketer} in 100 words',
    input_variables=['joke']
)

cricketer = input('Enter any cricketer name: ')
chain = RunnableParallel({
    'personal_bg': RunnableSequence(template1, model, parser),
    'prof_bg': RunnableSequence(template2 | model | parser)
})
result = chain.invoke({'cricketer': cricketer}) # Returned data is a dictionary
print(f'Personal background: {result['personal_bg']}\nProfessional background: {result['prof_bg']}')

chain.get_graph().draw_png('runnable_parallel_chain.png')