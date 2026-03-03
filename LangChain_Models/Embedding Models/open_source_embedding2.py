import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    
)

text = "Deep learning is a subset of machine learning."
vector = embeddings.embed_query(text)

print("Vector length:", len(vector))
print("First 5 values:", vector[:5])

# Multi-document embedding
documents = ["What is the current value of one dollar in inr?", "Who is Virat Kohli?", "Tell me something about deep learning", "Tell me about GATE examination"]

result = embeddings.embed_documents(documents)
for idx, vector in enumerate(result):
    print(f'Embedding vector {idx+1} -> (first five) -> {vector[:5]}')