import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    task='text-generation',
    pipeline_kwargs = dict(temperature=0.4),
)

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("Tell me something about deep learning in four to five sentences")
print(response.content)

# This will download the entire model on our local machine