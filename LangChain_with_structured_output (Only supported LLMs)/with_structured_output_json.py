import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
)

# Output schema
json_schema = None
with open('review.json', 'r') as file:
    json_schema = json.load(file)

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""
    Performance is reliable for everyday tasks like browsing, streaming, and social media.
""")

print(result) # python dictionary

# {'summary': 'Reliable performance for everyday tasks.', 'sentiment': 'positive', 'components_discussed': ['performance', 'everyday tasks', 'browsing', 'streaming', 'social media'], 'positive_sides': ['reliable performance', 'suitable for everyday tasks', 'good for browsing', 'good for streaming', 'good for social media'], 'negative_sides': None, 'rating': 8.5, 'recommendation': 'Buy the product'}