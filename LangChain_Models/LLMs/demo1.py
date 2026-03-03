import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq # Although this is a chat model, the code for llms looks somewhat similar, instead of ChatOpenAI we can have OpenAI, since it is paid, we cannot use it. 

# Load .env file to use the secret keys
load_dotenv()

# Get API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize model
llm = ChatGroq( # Similarly, here we would have called OpenAI
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
) # Initializing the model with the model-name and the API Key

model_response = llm.invoke("Who is Virat Kohli? Answer in two or three sentences.") # Calling the model on some human query (message)

print(model_response.content) # LLMs always sends string data as they are string in - string out models, so .content is not required to be applied on them