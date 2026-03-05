import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

chat_template = ChatPromptTemplate(
    messages=[
        ('system', 'You are a helpful {domain} expert. Answer carefully, cautiously and respectfully.'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', 'Explain me in simple terms and to the point in crisp about {topic}')
])

chat_history = []
try:
    with open('chat_history.txt', 'r') as file:
        chat_history = file.readlines()
except:
    chat_history = []

topic = input('Enter query you want to ask this doctor ai? ')

prompt = chat_template.invoke({
    'domain': 'doctor',
    'chat_history': chat_history,
    'topic': topic
})

result = model.invoke(prompt)
print(result.content)

new_history = []
new_history.append(HumanMessage(content=topic))
new_history.append(AIMessage(content=result.content))

with open('chat_history.txt', 'a') as file:
    for conv in new_history:
        file.write(str(conv) + "\n")