import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
)

# Output schema
class Review(BaseModel):
    summary: str = Field(description="Short summary of the review in one sentence")
    sentiment: Literal['Positive', 'Negative', 'Neutral'] = 'Neutral'
    rating: float = Field(default=0.00, ge=0.00, le=5.00)

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
    Performance is reliable for everyday tasks like browsing, streaming, and social media.
""")

print(result.model_dump_json())

# {"summary":"Performance is reliable for everyday tasks like browsing, streaming, and social media.","sentiment":"Positive","rating":4.0}