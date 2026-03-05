import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

messages = [
    SystemMessage(content='You are a helpful assistant. Answer correctly, politely and respectfully.'),
]
while True:
    user_input = input('You: ').lower()
    if user_input == 'exit':
        break
    messages.append(HumanMessage(content=user_input))
    result = model.invoke(messages)
    print(f'AI: {result.content}')
    messages.append(AIMessage(content=result.content))
    
print(messages)