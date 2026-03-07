import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token = huggingface_api_key,
)

model = ChatHuggingFace(llm = llm)
parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age, statistics and city of a cricketer\nFormat Instruction: {format_instruction}",
    input_variables=[],
    partial_variables={
        'format_instruction': parser.get_format_instructions() # this gives the format instructions
    } # this is called partial variables because this is not filled at runtime or by the user
)

prompt = template.invoke({}) # Nothing is required to be passed as we dont have any input variables
# print(prompt)

result = model.invoke(prompt).content
print(parser.parse(result))

# Method 2: Using chains
chain = template | model | parser
result = chain.invoke({})
print(result)

# JsonOutputParser cannot enforce a specific schema output from the model --> Drawback