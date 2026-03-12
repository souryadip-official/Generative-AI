import requests
import streamlit as st
from datetime import datetime
from news_fetcher import fetch_news
from llm_chain import generate_answer
from document_processor import convert_to_documents
from langchain_google_genai import ChatGoogleGenerativeAI
from vector_store import split_documents, create_vector_store, get_retriever

# Config
st.set_page_config(page_title="News Intelligence AI", page_icon="📰", layout="wide")

# Init state
if "keys_verified" not in st.session_state:
    st.session_state.keys_verified = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Validate APIs
def verify_apis(news_key, google_key):
    try:
        news_test = requests.get(f"https://newsapi.org/v2/everything?q=test&pageSize=1&apiKey={news_key}")
        if news_test.status_code != 200:
            return False, "News API Key invalid."
        
        test_llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", google_api_key=google_key)
        test_llm.invoke("Hi") 
        
        return True, "Keys verified!"
    except Exception as e:
        return False, f"API Error: {e}"

# Sidebar
with st.sidebar:
    st.title("⚙️ Setup")
    
    st.caption("Need an API key? Get them here:")
    st.link_button("🔗 Get News API Key", "https://newsapi.org/register", use_container_width=True)
    st.link_button("🔗 Get Google Gemini Key", "https://aistudio.google.com/app/apikey", use_container_width=True)
    st.divider()
    
    news_api_input = st.text_input("News API Key", type="password")
    google_api_input = st.text_input("Google Gemini API Key", type="password")
    
    if st.button("Verify Keys"):
        if not news_api_input or not google_api_input:
            st.warning("Enter both keys.")
        else:
            with st.spinner("Verifying..."):
                is_valid, message = verify_apis(news_api_input, google_api_input)
                if is_valid:
                    st.success(message)
                    st.session_state.keys_verified = True
                    st.session_state.news_api_key = news_api_input
                    st.session_state.google_api_key = google_api_input
                else:
                    st.error(message)
                    st.session_state.keys_verified = False

# Main UI
if not st.session_state.keys_verified:
    st.title("📰 AI News Assistant")
    st.info("👈 Verify API keys in the sidebar.")
else:
    # Date/Time in top right
    now = datetime.now().strftime("%A, %b %d, %Y | %I:%M %p")
    st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.9em;'>{now}</div>", unsafe_allow_html=True)
    
    st.title("📰 AI News Assistant")
    
    # Params
    with st.expander("🛠️ Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            news_topic = st.text_input("Topic", value="India")
        with col2:
            days_back = st.number_input("Days back", min_value=1, max_value=30, value=2)
        with col3:
            k_chunks = st.slider("Docs per query (k)", min_value=1, max_value=10, value=3)
            
        # Pipeline
        if st.button("Fetch & Process"):
            with st.status("Initializing RAG...", expanded=True) as status:
                try:
                    st.write("📡 Fetching news from server...")
                    articles = fetch_news(st.session_state.news_api_key, news_topic, days=days_back, page_size=30)
                    
                    st.write("🧹 Cleaning HTML...")
                    docs = convert_to_documents(articles)
                    
                    st.write("✂️ Splitting text...")
                    splits = split_documents(docs)
                    
                    st.write("🧠 Loading FAISS...")
                    vectorstore = create_vector_store(splits, st.session_state.google_api_key)
                    
                    st.write("🔍 Initializing retriever...")
                    st.session_state.retriever = get_retriever(vectorstore, k=k_chunks)
                    
                    status.update(label=f"Ready! {len(splits)} chunks loaded.", state="complete", expanded=False)
                    st.session_state.chat_history = []
                    
                except Exception as e:
                    status.update(label="Failed!", state="error", expanded=True)
                    st.error(f"Error: {e}")

    st.divider()

    # Chat UI
    if st.session_state.retriever:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_query := st.chat_input("Ask about the news..."):
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_query
            })
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Retrieve & Generate
                    retrieved_docs = st.session_state.retriever.invoke(user_query)
                    final_answer = generate_answer(user_query, retrieved_docs, st.session_state.google_api_key)
                    
                    st.markdown(final_answer)
                    
                    # Sources
                    with st.expander("📚 Sources"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            st.caption(f"**{doc.metadata.get('source', 'Unknown')}**: {doc.page_content[:150]}...")
                            
            st.session_state.chat_history.append({"role": "assistant", "content": final_answer})