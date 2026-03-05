import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def handle_response(click, question: str, style: str, length: str, insight: str):
    template = PromptTemplate(
        template="""
            You are an elite cricket analyst, commentator and historian with deep knowledge of international cricket, IPL, player statistics, match strategies and historical cricket events.

            The user has asked the following cricket question: {question_input}
            Response style: {style_input}
            Response length: {length_input}
            Insight depth: {insight_input}

            Instructions:
            1. Answer accurately using cricket knowledge, statistics and examples from real matches or players whenever possible.
            2. If the question is strategic, explain the cricketing logic behind the decision.
            3. If statistics are relevant, include them clearly.
            4. Use engaging commentary-like explanations when suitable.
            5. If information is uncertain or unavailable, do not hallucinate. Say "Insufficient reliable cricket information available".
            Refer to previous conversation in this session: {history_input}
        """,
        input_variables=['question_input', 'style_input', 'length_input', 'insight_input', 'history_input']
    )

    if question and click:
        prompt = template.invoke({
            'question_input': question,
            'style_input': style,
            'length_input': length,
            'insight_input': insight,
            'history_input': str(st.session_state.chat_history)
        })
        with st.spinner("Analyzing cricket data..."):
            response = model.invoke(prompt)

        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response.content
        })

        with st.chat_message("assistant", avatar="🏏"):
            st.markdown(response.content)

    elif not question and click:
        st.error('Please ask a cricket question first...')

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
model = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

st.set_page_config(page_title="CricketGPT", page_icon="🏏")
st.header('🏏 CricketGPT')

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_question' not in st.session_state:
    st.session_state.current_question = None

style = st.selectbox('Select response style', [
        'Commentary Style', 'Technical Analysis', 'Beginner Friendly', 'Statistical'
    ]
)
length = st.selectbox('Select response length', ['Short', 'Medium', 'Detailed'])
insight = st.selectbox('Select insight depth', ['Basic Explanation', 'Match Analysis', 'Deep Strategy'])

for message in st.session_state.chat_history:
    if message['role'] == 'user':
        with st.chat_message("user", avatar="🧑"):
            st.markdown(message['content'])
    else:
        with st.chat_message("assistant", avatar="🏏"):
            st.markdown(message['content'])

question = st.chat_input('Ask any cricket question...')
click = st.button('Analyze')

if question:
    st.session_state.current_question = question
    st.session_state.chat_history.append({
        'role': 'user',
        'content': question
    })
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

handle_response(
    click,
    st.session_state.current_question,
    style if style else 'Commentary Style',
    length if length else 'Short',
    insight if insight else 'Basic Explanation'
)