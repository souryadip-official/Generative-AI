# Objective: If feedback is of good sentiment say thank you, else ask for what to improve
import os
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-3-27b-it",
    huggingfacehub_api_token = huggingface_api_key,
)
model = ChatHuggingFace(llm = llm)

class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(description="Sentiment of the feedback")
strparser = StrOutputParser()
pydanticparser = PydanticOutputParser(pydantic_object=Feedback)

template1 = PromptTemplate(
    template="Find out the sentiment of the feedback: {feedback}\nFormat instruction: {format_instruction}",
    input_variables=['feedback'],
    partial_variables={
        'format_instruction': pydanticparser.get_format_instructions()
    }
)

template2 = PromptTemplate(
    template="This is a positive feedback from the user: {pos_feedback}. Write a simple appropriate reply message from our side to the user's positive feedback for their kind words. No need to give any options. Just generate a single reply to it.",
    input_variables=['pos_feedback']
)

template3 = PromptTemplate(
    template="This is a negative feedback from the user: {neg_feedback}. Write a simple appropriate reply message from our side to the user's negative feedback that we are apologetic and we will escalate this issue to a customer support executive. No need to give any options. Just generate a single reply to it.",
    input_variables=['neg_feedback']
)

feedback = input('Enter your feedback: ')
# Chains
classifier_chain = template1 | model | pydanticparser
# model_sentiment_response = classifier_chain.invoke({'feedback': feedback}).sentiment --> this is a pydantic object and we have directly implemented this flow in the RunnableBranch

branch_chain = RunnableBranch(
    (lambda model_sentiment_response: model_sentiment_response.sentiment == 'Positive', template2 | model | strparser), # format: (condition, chain to execute upon truth of the condition)
    (lambda model_sentiment_response: model_sentiment_response.sentiment == 'Negative', template3 | model | strparser),
    RunnableLambda(lambda x: "Could not find sentiment") # Doing simply this, lambda x: "Could not find sentiment", won't work because we are executing chains over here not statements. So we need to convert this to a runnable. RunnableLambda converts a lambda function into a runnable
)

final_chain = classifier_chain | branch_chain
result = final_chain.invoke({'feedback': feedback})
print(result)

# To visualize the chain
print(final_chain.get_graph().print_ascii())