import os
import warnings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
load_dotenv()
warnings.filterwarnings('ignore')

groq_api_key = os.getenv('GROQ_API_KEY')
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)
strparser = StrOutputParser()

class Feedback(BaseModel):
    ftype: Literal['General Query', 'Refund Request', 'Complaint Request'] = Field("Feedback type of a customer\'s feedback")
pydanticparser = PydanticOutputParser(pydantic_object=Feedback)

# Our job is to take a customer feedback, find out the type of feedback by an llm like a general query, complain report or refund request and accordingly give our response

template1 = PromptTemplate(
    template='You are a customer feedback analyzer. Find the feedback type of the given user feedback \"{feedback}\"\nFormat instruction: {format_instruction}',
    input_variables=['feedback'],
    partial_variables={
        'format_instruction': pydanticparser.get_format_instructions()
    }
)

template2 = PromptTemplate(
    template="""
        You are a customer support representative.
        A customer sent the following message: {feedback}
        Write a polite and helpful response directly to the customer.
        Guidelines:
        - Address the customer naturally (do not use placeholders like [Customer Name]).
        - Do not mention roles like "General Query Head".
        - Do not include template placeholders like [Company Name] or [Your Name] or [Customer Name].
        - Respond like a real support agent in 4-6 sentences.
    """,
    input_variables=['feedback']
)

template3 = PromptTemplate(
    template="""
        You are a customer support representative handling refund requests.
        Customer message: {feedback}
        Write a professional response that:
        - Acknowledges the issue
        - Explains that the refund process will be handled
        - Asks for necessary details if needed
        Guidelines:
        - Speak directly to the customer.
        - Do not include placeholders like [Customer Name] or [Company Name] or [Your Name].
        - Keep the response clear, polite, and reassuring (4-6 sentences).
    """,
    input_variables=['feedback']
)

template4 = PromptTemplate(
    template="""
        You are a customer support representative handling customer complaints.
        Customer message: {feedback}
        Write a response that:
        - Apologizes for the inconvenience
        - Shows empathy
        - Assures the customer that the issue will be investigated
        Guidelines:
        - Speak directly to the customer.
        - Do not use placeholders like [Customer Name] or [Company Name] or [Your Name].
        - Do not mention your role or department.
        - Keep the response polite and professional (4-6 sentences).
    """,
    input_variables=['feedback']
)

feedback = input('Enter your feedback: ')
chain1 = RunnableParallel({
    'f': RunnableSequence(template1, model, pydanticparser),
    'feedback': RunnablePassthrough()
})
chain2 = RunnableParallel({
    'feedback_handler': RunnableBranch(
    (lambda d: d['f'].ftype == 'General Query', 
        RunnableSequence(
            RunnableLambda(lambda d: {'feedback': d['feedback']}),
            template2, model, strparser)),
    (lambda d: d['f'].ftype == 'Refund Request',
        RunnableSequence(
            RunnableLambda(lambda d: {'feedback': d['feedback']}),
            template3, model, strparser)),
    (lambda d: d['f'].ftype == 'Complaint Request',
        RunnableSequence(
            RunnableLambda(lambda d: {'feedback': d['feedback']}),
            template4, model, strparser)),
    RunnableLambda(lambda x: "No feedback type found")
),
    'feedback_type': RunnableLambda(lambda d: d['f'].ftype)})

final_chain = RunnableSequence(chain1, chain2)
result = final_chain.invoke({'feedback': feedback})
print(f"Feedback type: {result['feedback_type']}\n\n{result['feedback_handler']}")

final_chain.get_graph().draw_png('runnable_branch_chain_02.png')