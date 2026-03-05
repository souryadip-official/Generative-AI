import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# print(st.__version__)
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
model = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    google_api_key=google_api_key,
)

st.header('Research Assistant 🧠')
user_input = str(st.text_input('Please enter your query...')).strip()
click = st.button('Search')
if user_input and click:
    response = model.invoke(user_input)
    st.markdown(response.content)
elif not user_input and click:
    st.error('Please write a query first...')