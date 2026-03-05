import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, load_prompt

def handle_response(click, rpn: str, es: str, el: str):
    template = load_prompt('prompt.json')
    if rpn and click:
        prompt = template.invoke({
            'paper_input': rpn,
            'style_input': es,
            'length_input': el
        })
        response = model.invoke(prompt)
        st.markdown(response.content)
    elif not rpn and click:
        st.error('Please write a resrarch paper name first...')

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
model = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    google_api_key=google_api_key,
)

st.header('Research Assistant 🧠')
research_paper_name = str(st.text_input('Enter research paper name')).strip()
explaination_style = st.selectbox('Select explanation style', [
    'Beginner-Friendly', 'Technical', 'Code-Oriented', 'Mathematical'
])
length = st.selectbox(f'Select explanation length', [
    "Short (1-2 paragraphs)", "Medium (4-5 paragraphs)", "Long (In depth explanation)"
])

click = st.button('Submit request')
handle_response(click, research_paper_name if research_paper_name else None, explaination_style if explaination_style else 'Beginner-Friendly', length if length else 'Short (1-2 paragraphs)')