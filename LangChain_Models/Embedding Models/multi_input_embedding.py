import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=google_api_key,
    output_dimensionality=4, # length of the vector
)

documents = ["What is the current value of one dollar in inr?", "Who is Virat Kohli?", "Tell me something about deep learning", "Tell me about GATE examination"]

result = embeddings.embed_documents(documents)
for idx, vector in enumerate(result):
    print(f'Embedding vector {idx}: {vector}')