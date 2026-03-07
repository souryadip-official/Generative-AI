import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict
load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # this model supports structured output
    google_api_key=google_api_key,
)

# Output schema
class Review(TypedDict):
    summary: str
    sentiment: str
    rating: int

structured_model = model.with_structured_output(Review) # This internally generates a system prompt for the model

result = structured_model.invoke("""
    Absolutely loving the display on the Motorola Edge 50 Pro. The 144Hz
    screen feels extremely smooth and scrolling through apps feels
    buttery. Probably the best display I've used in this price range.
""")

print(result) # No need to print content as the result will be exactly as the schema we defined
# Sample output: {'summary': 'The display on the Motorola Edge 50 Pro is excellent, offering a very smooth 144Hz experience, which is considered the best in its price range.', 'sentiment': 'positive', 'rating': 5}

# Since this is a dictionary, we can seperately fetch keys
print(result['summary'])
print(result['sentiment'])

# This code works only if the model we are using is capable of generating structured output