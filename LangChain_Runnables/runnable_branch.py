import os
import warnings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
# RunnableBranch is the if-else conditions in the world of LangChain.
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

# Our job is to generate a joke, explanation of it, print number of words in the joke and if the joke length is greater than 50 words, print joke_size = 'long', if >35, 'medium' else 'short'

def count_words(joke: str):
    return len(joke.strip().split())

topic = input('Enter any topic: ')
joke_gen_chain = RunnableSequence(template1, model, parser)
joke_explain_with_orig_joke = RunnableParallel({
    'joke_explain': RunnableSequence(template2, model, parser),
    'original_joke': RunnablePassthrough(),
    'words': RunnableLambda(count_words),
}) # this will return a dictionary
joke_size_chain = RunnableBranch(
    (lambda d: d['words'] > 50, RunnableLambda(lambda d: {**d, 'joke_size': 'long'})),
    (lambda d: d['words'] > 35, RunnableLambda(lambda d: {**d, 'joke_size': 'medium'})),
    RunnableLambda(lambda d: {**d, 'joke_size': 'short'}) # Dictionary unpacking and new insertion
)
final_chain = RunnableSequence(joke_gen_chain, joke_explain_with_orig_joke, joke_size_chain)
result = final_chain.invoke({'topic': topic})
print(f"Joke: {result['original_joke']}\n\n\nExplanation: {result['joke_explain']}\n\nNumber of words in the joke: {result['words']}\n\nJoke size: {result['joke_size']}")

final_chain.get_graph().draw_png('runnable_branch_chain.png')