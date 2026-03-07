# PydanticOutputParser is a structured output parser in LangChain that uses pydantic models to enforce schema validation when processing LLM responses
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token = huggingface_api_key,
)

model = ChatHuggingFace(llm = llm)

class Cricketer(BaseModel):
    name: str = Field(description='Name of the international cricketer')
    age: int = Field(description='Age of the international cricketer', ge=18, lt=45)
    country: str = Field(description='Country for which the international cricketer plays for')
    city: str = Field(description='City of the international cricketer from which they belong to')
    contact_email: Optional[EmailStr] = Field(default=None, description="Contact email of the international cricketer to reach out")
    
parser = PydanticOutputParser(pydantic_object=Cricketer)
template = PromptTemplate(
    template="Give me the name, age, statistics and city of any random international cricketer belonging to the country {country}\nFormat Instruction: {format_instruction}",
    input_variables=['country'],
    partial_variables={
        'format_instruction': parser.get_format_instructions()
    }
)

country = input('Enter any country: ')
prompt = template.invoke({
    'country': country
})
print(prompt)
result = model.invoke(prompt).content
print(parser.parse(result))

# Approach 2: Using chains
chain = template | model | parser
result = chain.invoke({
    'country': country
})
print(result)