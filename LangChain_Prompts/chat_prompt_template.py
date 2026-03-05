import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

chat_template = ChatPromptTemplate(
    messages=[
        ('system', 'You are a helpful {domain} expert. Answer carefully, cautiously and respectfully.'),
        ('human', 'Explain me in simple terms and to the point in crisp about {topic}')
])
prompt = chat_template.invoke({
    'domain': 'doctor',
    'topic': 'common cold'
})
print(model.invoke(prompt).content)