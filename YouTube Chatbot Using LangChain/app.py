import streamlit as st
from datetime import datetime
from huggingface_hub import HfApi
from llm_chain import generate_answer
from fetch_transcript import fetch_transcript
from transcript_processor import split_text, create_vector_store, get_retriever

st.set_page_config(page_title="YouTube AI Tutor", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.verify-disabled{
    opacity:0.45;
    pointer-events:none;
    filter:grayscale(60%);
}
button[kind="primary"]{
    background-color:#28a745 !important;
    border-color:#28a745 !important;
    color:white !important;
}
button[disabled]{
    background-color:#7fbf7f !important;
    border-color:#7fbf7f !important;
    color:white !important;
    opacity:1 !important;
}
button[kind="secondary"]{
    background-color:#dc3545 !important;
    border-color:#dc3545 !important;
    color:white !important;
}
</style>
""", unsafe_allow_html=True)

# Session initialization
if "keys_verified" not in st.session_state:
    st.session_state.keys_verified=False
if "retriever" not in st.session_state:
    st.session_state.retriever=None
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]
if "video_url" not in st.session_state:
    st.session_state.video_url=None

# Access token verification
def verify_hf_token(token):
    try:
        api=HfApi()
        api.whoami(token=token)
        return True,"Key verified!"
    except:
        return False,"Invalid HuggingFace API key."

# Sidebar interface (key and chatbot)
with st.sidebar:
    st.title("⚙️ Setup")
    st.caption("Need a HuggingFace key?")
    st.link_button(
        "🔗 Get HuggingFace API Key",
        "https://huggingface.co/settings/tokens",
        use_container_width=True
    )
    st.divider()

    # User key
    hf_input=st.text_input("HuggingFace API Key",type="password")
    if not st.session_state.keys_verified:
        if st.button("Connect",type="primary",use_container_width=True):
            if not hf_input:
                st.warning("Enter API key.")
            else:
                with st.spinner("🔍 Verifying HuggingFace key..."):
                    ok,msg=verify_hf_token(hf_input)
                    if ok:
                        st.success(msg)
                        st.session_state.keys_verified=True
                        st.session_state.hf_api_key=hf_input
                        st.rerun()
                    else:
                        st.error(msg)
    else:
        st.button("Connected",disabled=True,use_container_width=True, type='primary')
        if st.button("Disconnect",type="secondary",use_container_width=True):
            # refresh and reinitialization
            st.session_state.keys_verified=False
            st.session_state.retriever=None
            st.session_state.chat_history=[]
            st.session_state.video_url=None
            st.rerun()
            
    st.divider()

    # Chatbot interface
    if st.session_state.video_url:
        st.title("💬 Chat")
        chat_container=st.container()
        with chat_container:
            if len(st.session_state.chat_history)==0:
                st.chat_message("assistant").write("Hello! You can ask me anything about the video.")
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # User query           
        if user_query:=st.chat_input("Ask about the video..."):
            st.session_state.chat_history.append({
                "role":"user",
                "content":user_query
            })
            final_answer=""
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)
                with st.chat_message("assistant"):
                    with st.spinner("🤖 Thinking..."):
                        retrieved_docs=st.session_state.retriever.invoke(user_query)
                        final_answer=generate_answer(
                            user_query,
                            retrieved_docs,
                            st.session_state.hf_api_key
                        )
                        st.markdown(final_answer)
                        with st.expander("📚 Transcript Context"):
                            for i,doc in enumerate(retrieved_docs,1):
                                st.caption(f"Chunk {i}")
                                st.write(doc.page_content)
                                st.markdown("---")
            st.session_state.chat_history.append({
                "role":"assistant",
                "content":final_answer
            })

# Right side screen
if not st.session_state.keys_verified:
    st.title("🧠 YouTube AI Learning Assistant")
    st.info("👈 Verify your HuggingFace API key in the sidebar to begin.")
else:
    now=datetime.now().strftime("%A, %b %d, %Y | %I:%M %p")
    st.markdown(
        f"<div style='text-align:right;color:gray;font-size:0.9em'>{now}</div>",
        unsafe_allow_html=True
    )
    st.title("🧠 YouTube AI Tutor")
    with st.expander("📺 Video",expanded=True):
        disabled_state = st.session_state.video_url is not None
        video_url=st.text_input(
            "Paste YouTube video link",
            placeholder="https://www.youtube.com/watch?v=...",
            disabled=disabled_state
        )
        st.caption("Open YouTube -> Click Share -> Copy Link -> Paste here")
        if not disabled_state:
            if st.button("Process Video"):
                if not video_url:
                    st.warning("Paste a YouTube link first.")
                else:
                    with st.status("Initializing RAG...",expanded=True) as status:
                        try:
                            st.write("📥 Fetching transcript")
                            transcript=fetch_transcript(video_url)

                            st.write("✂️ Splitting transcript")
                            docs=split_text(transcript)

                            st.write("🧠 Building vector store")
                            vectorstore=create_vector_store(
                                docs,
                                st.session_state.hf_api_key
                            )
                            st.write("🔍 Preparing retriever")
                            st.session_state.retriever=get_retriever(
                                st.session_state.hf_api_key,
                                vectorstore,
                                k=4
                            )
                            st.session_state.video_url=video_url
                            st.session_state.chat_history=[]
                            status.update(
                                label="Ready! Video processed.",
                                state="complete",
                                expanded=False
                            )
                            st.rerun()

                        except Exception as e:

                            status.update(
                                label="Processing failed!",
                                state="error",
                                expanded=True
                            )

                            st.error(f"Error: {e}")

        else:
            if st.button("New Video"):
                st.session_state.video_url=None
                st.session_state.retriever=None
                st.session_state.chat_history=[]
                st.rerun()

    st.divider()

    if st.session_state.video_url:
        st.video(st.session_state.video_url)