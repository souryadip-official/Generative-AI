import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal # Optional: What if we have some key that is not mandatorily needed for an output. For example in a product review system, if we have a good side and a bad side, then it is not mandatory to have both. So they must be optional features, Literal: Say we want output from the model in a range of few options, for example if we are dealing with a cricket match system, a match type can either be T20, ODI or Test. So the outputs from the model are limited
load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
)

# Output schema
class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "Sentiment of the review like positive, negative, neutral"]
    components_discussed: Annotated[list[str], "Write all the components of the product that is talked about in the review"]
    positive_sides: Annotated[Optional[list[str]], "Write the positive sides of the product in the review"]
    negative_sides: Annotated[Optional[list[str]], "Write the negative sides of the product in the review"]
    rating: Annotated[float, "A rating of the review between 0.00 and 10.00"]
    recommendation: Annotated[Literal['Buy the product', 'Don\'t buy the product', 'No comments'], "Write the recommendation of the user to buy the product or not"]
    

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
    Performance is reliable for everyday tasks like browsing, streaming, and social media.
""")

print(result)

# {'summary': 'The performance is reliable for everyday tasks such as browsing, streaming, and social media.', 'sentiment': 'Positive', 'components_discussed': ['Performance'], 'positive_sides': ['Reliable performance for everyday tasks', 'Good for browsing', 'Good for streaming', 'Good for social media'], 'negative_sides': [], 'rating': 4, 'recommendation': 'Buy the product'}