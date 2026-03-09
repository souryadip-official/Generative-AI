import os
import warnings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
# RunnableLambda convers a function into a runnable.
load_dotenv()
warnings.filterwarnings('ignore')

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
    template="Explain the joke in simple words. Joke is \"{joke}\"",
    input_variables=['joke']
)

# Our job is to generate a joke, explanation of it and print number of words in the joke
def count_words(joke: str):
    return len(joke.strip().split())

topic = input('Enter any topic: ')
joke_gen_chain = RunnableSequence(template1, model, parser)
joke_explain_with_orig_joke = RunnableParallel({
    'joke_explain': RunnableSequence(template2, model, parser),
    'original_joke': RunnablePassthrough(),
    'words': RunnableLambda(count_words)
})
final_chain = RunnableSequence(joke_gen_chain, joke_explain_with_orig_joke)
result = final_chain.invoke({'topic': topic})
print(f'Joke: {result['original_joke']}\n\n\nExplanation: {result['joke_explain']}\n\nNumber of words in the joke: {result['words']}')

final_chain.get_graph().draw_png('runnable_lambda_chain.png')