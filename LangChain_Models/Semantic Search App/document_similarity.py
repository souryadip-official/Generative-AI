import os
import numpy as np
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=google_api_key,
    output_dimensionality=300, # length of the vector
)

documents = [
    "Virat Kohli is one of the greatest modern-day cricketers, known for his aggressive batting style, exceptional consistency across all formats, and inspirational leadership for India in international cricket.",
    "Rohit Sharma is an elegant opening batsman famous for his effortless stroke play, record-breaking double centuries in ODIs, and successful captaincy of the Indian cricket team.",
    "Sachin Tendulkar is widely regarded as the 'God of Cricket', holding numerous batting records including 100 international centuries and inspiring generations of cricketers worldwide.",
    "Rinku Singh is a dynamic middle-order batsman recognized for his explosive finishing abilities, particularly highlighted by his remarkable last-over performances in the IPL.",
    "Jasprit Bumrah is a world-class fast bowler known for his unique action, deadly yorkers, and exceptional ability to perform under pressure in all formats of cricket."
]

user_query = "why is rinku singh becoming popular?"
vector_database = embeddings.embed_documents(documents)
user_query_embedding = embeddings.embed_query(user_query)

similarity_score = cosine_similarity([user_query_embedding], vector_database) # First param whom to compare, second param with what to compare
similarity_score = similarity_score[0]
most_similar = np.argmax(similarity_score)
print(f'The user query, \"{user_query}\" looks somewhat similar to document {most_similar+1} which has content {documents[most_similar]}')