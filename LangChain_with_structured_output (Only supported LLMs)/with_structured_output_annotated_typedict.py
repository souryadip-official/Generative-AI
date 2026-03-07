import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
)

# Output schema
class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"] # Explicitly describe what we exactly want
    sentiment: Annotated[str, "Sentiment of the review like positive, negative, neutral"]
    components_discussed: Annotated[list[str], "Write all the components of the product that is talked about in the review"]
    rating: Annotated[float, "A rating of the review between 0.00 and 10.00"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
    I have been using the Motorola Edge 50 Pro for a few weeks now, and my overall experience has been somewhat mixed, though not necessarily in a bad way. The display is definitely one of the highlights of the device. The 144Hz OLED panel feels extremely smooth when scrolling or navigating through apps, and the colors look vibrant while watching videos. At the same time, the curved edges that make the display look premium can occasionally lead to accidental touches, which slightly reduces the practicality of the otherwise excellent screen. Still, the visual experience remains enjoyable for most everyday tasks.

    In terms of performance, the phone generally feels fast and responsive. Apps open quickly and multitasking works well most of the time. However, during longer gaming sessions or heavy usage, the phone sometimes becomes warm, which doesn't necessarily affect performance significantly but is noticeable enough to mention. For normal activities like browsing, streaming, and social media, the phone performs smoothly, though it doesn't always feel dramatically faster than other phones in the same price range.

    The camera system is another area where the experience can vary depending on the situation. In daylight, photos tend to look sharp with good color reproduction, and the images are easily good enough for sharing online. On the other hand, low-light photography can be inconsistent. Sometimes the results are impressive, while other times the images appear slightly softer or noisier than expected. Video recording is decent and usable, though stabilization is not always as strong as what some competing devices offer.

    Battery life is generally adequate for daily use. With moderate usage the phone usually lasts through a full day, which is convenient. However, heavier usage such as gaming, camera usage, or constant 5G connectivity can make the battery drain faster than expected. The fast charging helps compensate for this because the phone charges very quickly, which makes short charging sessions quite effective.

    The design of the device is another interesting aspect. It feels slim and premium in the hand, and the vegan leather back option adds a distinctive look while also improving grip. At the same time, the curved display and slim form factor may not appeal to everyone, especially for users who prefer completely flat screens for durability and easier screen protector installation.

    On the software side, the near-stock Android experience is clean and simple, which many users will appreciate because there are very few unnecessary apps or interface changes. The interface feels smooth and easy to use, and Motorola's gesture features are surprisingly practical once you start using them regularly. However, software updates from Motorola have historically been slower compared to some other brands, so long-term support might be something potential buyers consider before purchasing.

    Overall, the Motorola Edge 50 Pro feels like a well-balanced smartphone that performs reliably in many areas without being completely dominant in any single category. It offers a good display, fast charging, and clean software, while also having a few trade-offs like occasional heating, mixed low-light camera results, and average battery endurance.
""")

print(result)

# {'summary': 'The Motorola Edge 50 Pro offers a mixed but well-balanced experience. It features an excellent 144Hz OLED display, generally fast performance, and a clean near-stock Android software experience with practical gestures. The design is premium, with a vegan leather option and fast charging is a significant plus. However, it has trade-offs including occasional heating during heavy use, inconsistent low-light camera performance, average video stabilization, and moderate battery endurance under heavy load. Software updates have historically been slow.', 'sentiment': 'balanced', 'components_discussed': ['display', 'performance', 'camera system', 'battery life', 'design', 'software'], 'rating': 3.8}