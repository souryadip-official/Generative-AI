# Open source
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

text = "Deep learning is a subset of machine learning."
vector = embeddings.embed_query(text)

print("Vector length:", len(vector))
print("First 5 values:", vector[:5])

# To download the model in our local machine