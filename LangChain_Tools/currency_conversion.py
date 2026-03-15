import os
import warnings
warnings.filterwarnings('ignore')

import requests
from typing import Annotated
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
exchange_rate_api_key = os.getenv('EXCHANGE_RATE_API_KEY')

@tool
def fetch_conversion_rate(base_curr: str, target_curr: str) -> float:
    """Takes 2 currencies and finds their currency conversion rate"""
    result = requests.get(f"https://v6.exchangerate-api.com/v6/{exchange_rate_api_key}/pair/{base_curr}/{target_curr}").json()
    return result["conversion_rate"]

@tool
def convert(base_currency_value: float, rate: Annotated[float, InjectedToolArg]) -> float:
    """Given a currency conversion rate this function calculates the target currency value from a given base currency value"""
    return rate * base_currency_value

rate = None
def execute_tool(tool_response):
    global rate
    if tool_response['name'] == 'fetch_conversion_rate':
        fetch_response = fetch_conversion_rate.invoke(tool_response)
        rate = float(fetch_response.content)
        return fetch_response
    elif tool_response['name'] == 'convert':
        tool_response['args']['rate'] = rate
        fetch_response = convert.invoke(tool_response)
        return fetch_response
        
huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=huggingfacehub_api_token,
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

model = model.bind_tools([fetch_conversion_rate, convert])

user_query = input('Enter conversion query...')
messages = [HumanMessage(user_query)]

ai_message1 = model.invoke(messages)
messages.append(ai_message1)

tool_call1 = ai_message1.tool_calls[0]
fetch_response1 = execute_tool(tool_call1)
messages.append(fetch_response1)

ai_message2 = model.invoke(messages)
tool_call2 = ai_message2.tool_calls[0]
fetch_response2 = execute_tool(tool_call2)
messages.append(fetch_response2)

template = PromptTemplate(
    template = """
    Given to you a sequence of interaction between you and user,
    write a good crafted and respectful response message based on the conversation.
    Conversation: {conversation}""",
    input_variables=['conversation']
)
chain = template | model | parser
print(chain.invoke({
    'conversation': messages
}))