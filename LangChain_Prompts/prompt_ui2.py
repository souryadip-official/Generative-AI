import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

def handle_response(click, rpn: str, es: str, el: str):
    template = PromptTemplate(
        template="""
            You are an experienced and accurate and trusted research professional.
            Please summarize and give insights about the research paper titled {paper_input} with
            the following specifications:
            Explanation style: {style_input}
            Explanation length: {length_input}
            1. Mathematical details: Include relevant mathematical equations if present in the paper and
            explain the mathematical concepts using simple, intuitive code snippets wherever applicable and break complex equations into smaller and slowly converge them into the bigger equation so that the user can actually understand the formulation.
            2. Analogies: Use relatable real-world analogies that may sound funny to the user so that they can enjoy their learning time and understand complex ideas very easily.
            If certain information is not available in the paper or the research paper user has given has no sufficient information or any sort of thing that is unclear to you, dont hallucinate, just reply with "Insufficient information available..." without random guessing. Make sure that the report you give is factually correct, accurate, clear and aligned with the provided style and length.
        """,
        input_variables=['paper_input', 'style_input', 'length_input']
    )
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