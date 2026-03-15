import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
# Tool creation
@tool
def multiply(a: float, b: float) -> float:
    """Performs multiplication of two numbers a and b"""
    return a * b

google_api_key = os.getenv('GOOGLE_API_KEY')
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
)
parser = StrOutputParser()

model = model.bind_tools([multiply])

template = PromptTemplate(
    template = """
        Given the user query: {query}
        Solve this query respectfully.
    """,
    input_variables=['query']
)
chain = template | model
response = chain.invoke({
    "query": input('Type the question...')
})
tool_response = response.tool_calls[0]
print(multiply.invoke(tool_response['args']))
tool_response = multiply.invoke(tool_response)

template2 = PromptTemplate(
    template = """
    This my tool's response.
    {response}
    Arrange this to form the final sentenced response.""",
    input_variables=['response']
)

chain2 = template2 | model | parser
print(chain2.invoke({
    'response': tool_response
}))