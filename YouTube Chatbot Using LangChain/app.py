import streamlit as st
import time
from datetime import datetime
from huggingface_hub import HfApi
from llm_chain import generate_answer
from fetch_transcript import fetch_transcript
from transcript_processor import split_text, create_vector_store, get_retriever

st.set_page_config(
    page_title="AskTube AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Syne:wght@600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1200px; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: #0A0A0F !important;
    background-image:
        radial-gradient(ellipse 80vw 60vh at 10% 0%,   rgba(108,99,255,0.18) 0%, transparent 55%),
        radial-gradient(ellipse 60vw 50vh at 90% 10%,  rgba(20,184,166,0.12) 0%, transparent 50%),
        radial-gradient(ellipse 50vw 40vh at 50% 90%,  rgba(244,114,182,0.08) 0%, transparent 50%),
        radial-gradient(ellipse 40vw 30vh at 80% 80%,  rgba(251,191,36,0.06) 0%, transparent 50%) !important;
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: rgba(14,14,22,0.70) !important;
    backdrop-filter: blur(32px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(32px) saturate(180%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] > div { background: transparent !important; }
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] label {
    color: rgba(255,255,255,0.40) !important;
    font-size: 11px !important; font-weight: 600 !important; letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] p { color: rgba(255,255,255,0.70) !important; font-size: 13px !important; }

[data-testid="stSidebar"] input[type="password"],
[data-testid="stSidebar"] input[type="text"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 12px !important; color: #fff !important;
    font-size: 13px !important; transition: all 0.2s !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: rgba(108,99,255,0.60) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.15), 0 0 20px rgba(108,99,255,0.10) !important;
    background: rgba(108,99,255,0.08) !important;
}
[data-testid="stSidebar"] input::placeholder { color: rgba(255,255,255,0.25) !important; }

[data-testid="stSidebar"] [data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
}
[data-testid="stSidebar"] [data-testid="stChatInput"] textarea {
    color: rgba(255,255,255,0.85) !important;
}

/* Chatbox Specific Styling */
[data-testid="stChatMessage"] { background: transparent !important; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { font-size: 15px !important; line-height: 1.6 !important; }

/* Inline code only */
[data-testid="stChatMessage"] code:not(pre code) { 
    font-family: 'Consolas', 'Lucida Console', monospace !important; 
    font-size: 14px !important; 
    color: #F992C8 !important; 
    background: rgba(255,255,255,0.08) !important; 
    padding: 2px 6px !important; 
    border-radius: 6px !important; 
}

/* Code block container */
[data-testid="stChatMessage"] pre { 
    background: #0D0D12 !important; /* Very dark slate for contrast */
    padding: 16px !important; 
    border-radius: 12px !important; 
    border: 1px solid rgba(255,255,255,0.1) !important; 
    overflow-x: auto !important; 
    margin: 10px 0 !important; 
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important; 
}

/* Let syntax highlighting colors apply organically inside pre code */
[data-testid="stChatMessage"] pre code { 
    font-family: 'Consolas', 'Lucida Console', monospace !important; 
    font-size: 14px !important; 
    background: transparent !important; 
    padding: 0 !important; 
    border-radius: 0 !important; 
}

.katex { font-size: 1.1em !important; }

/* Global overrides (excluding spans to protect syntax highlighting) */
.stApp p, .stApp div, .stApp label { color: rgba(255,255,255,0.85) !important; }
.stApp h1, .stApp h2, .stApp h3 { color: #fff !important; font-family: 'Syne', sans-serif !important; }
.stApp .stCaption { color: rgba(255,255,255,0.40) !important; font-size: 12px !important; }

button[kind="primary"] {
    background: linear-gradient(135deg, #6C63FF 0%, #8B83FF 50%, #6C63FF 100%) !important;
    background-size: 200% 200% !important;
    border: none !important; color: #fff !important;
    border-radius: 14px !important; font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: 14px !important;
    padding: 0.6rem 1.8rem !important;
    box-shadow: 0 0 24px rgba(108,99,255,0.40), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    transition: all 0.25s !important; letter-spacing: 0.3px !important;
}
button[kind="primary"]:hover {
    box-shadow: 0 0 40px rgba(108,99,255,0.60), inset 0 1px 0 rgba(255,255,255,0.20) !important;
    transform: translateY(-2px) !important;
}
button[kind="primary"]:disabled { opacity: 0.45 !important; transform: none !important; }

button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: rgba(255,255,255,0.70) !important; border-radius: 14px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
    transition: all 0.2s !important;
}
button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.20) !important;
    color: #fff !important;
}

[data-testid="stLinkButton"] a {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: rgba(255,255,255,0.80) !important; border-radius: 12px !important;
    font-size: 13px !important; font-weight: 500 !important;
    transition: all 0.2s !important;
}
[data-testid="stLinkButton"] a:hover {
    background: rgba(255,255,255,0.10) !important; color: #fff !important;
}

[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important; color: #fff !important;
    font-size: 14px !important; transition: all 0.2s !important;
    padding: 12px 16px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(108,99,255,0.55) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.15), 0 0 20px rgba(108,99,255,0.10) !important;
    background: rgba(108,99,255,0.08) !important;
}
[data-testid="stTextInput"] input::placeholder { color: rgba(255,255,255,0.25) !important; }
[data-testid="stTextInput"] label { color: rgba(255,255,255,0.45) !important; font-size: 12px !important; font-weight:500 !important; }

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important; padding: 4px !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important; border-radius: 10px !important;
    color: rgba(255,255,255,0.45) !important; border: none !important;
    font-size: 13px !important; font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important; padding: 7px 18px !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(108,99,255,0.25) !important;
    color: #fff !important;
    box-shadow: 0 0 12px rgba(108,99,255,0.20) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
}
details summary { color: rgba(255,255,255,0.60) !important; font-size: 13px !important; font-weight: 500 !important; }

[data-testid="stAlert"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important; border: 1px solid !important;
}
[data-testid="stAlert"] * { color: rgba(255,255,255,0.85) !important; }

[data-testid="stVideo"] iframe {
    border-radius: 18px !important;
    box-shadow: 0 8px 48px rgba(0,0,0,0.60), 0 0 0 1px rgba(255,255,255,0.06) !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.30); border-radius: 99px; }

hr { border-color: rgba(255,255,255,0.06) !important; }

.gc {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.gc::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
}
.gc:hover {
    border-color: rgba(255,255,255,0.16);
    box-shadow: 0 8px 40px rgba(0,0,0,0.30);
}

.gc-v { background: rgba(108,99,255,0.08); border-color: rgba(108,99,255,0.22); }
.gc-v:hover { box-shadow: 0 8px 40px rgba(108,99,255,0.15); }
.gc-t { background: rgba(20,184,166,0.08); border-color: rgba(20,184,166,0.22); }
.gc-t:hover { box-shadow: 0 8px 40px rgba(20,184,166,0.15); }
.gc-p { background: rgba(244,114,182,0.08); border-color: rgba(244,114,182,0.22); }
.gc-p:hover { box-shadow: 0 8px 40px rgba(244,114,182,0.15); }

.pill {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.3px;
    padding: 4px 12px; border-radius: 99px;
}
.pv { background: rgba(108,99,255,0.20); color: #A09CF8; border: 1px solid rgba(108,99,255,0.35); }
.pt { background: rgba(20,184,166,0.18); color: #5ECFC4; border: 1px solid rgba(20,184,166,0.35); }
.pp { background: rgba(244,114,182,0.18); color: #F992C8; border: 1px solid rgba(244,114,182,0.35); }
.py { background: rgba(251,191,36,0.18);  color: #FBD96B; border: 1px solid rgba(251,191,36,0.35); }
.ps { background: rgba(99,179,237,0.18);  color: #87C9F0; border: 1px solid rgba(99,179,237,0.35); }

.stat-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin: 18px 0; }
.stat-tile {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 18px 16px; text-align: center;
    transition: all 0.2s;
}
.stat-tile:hover { background: rgba(255,255,255,0.07); border-color: rgba(255,255,255,0.14); }
.stat-n { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; line-height: 1; margin-bottom: 5px; }
.stat-l { font-size: 11px; color: rgba(255,255,255,0.40); font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; }

.ins {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 16px 18px; margin-bottom: 10px;
    transition: all 0.2s; position: relative; overflow: hidden;
}
.ins::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,0.10),transparent); }
.ins:hover { background: rgba(255,255,255,0.07); border-color: rgba(255,255,255,0.14); transform: translateY(-2px); }
.ins-h { display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; }
.ins-b { font-size: 13px; color: rgba(255,255,255,0.65); line-height: 1.7; }

.sc-row { display:flex; align-items:center; gap:12px; padding:8px 0; }
.sc-rk { font-size:12px; font-weight:700; color:rgba(255,255,255,0.30); width:22px; }
.sc-bg { flex:1; height:5px; background:rgba(255,255,255,0.07); border-radius:99px; overflow:hidden; }
.sc-b  { height:100%; border-radius:99px; }
.sc-v  { font-size:12px; font-weight:600; min-width:34px; text-align:right; }

.pipe { display:flex; align-items:center; padding:14px 0; margin-bottom:4px; }
.ps-step { display:flex; align-items:center; gap:8px; }
.ps-dot {
    width:30px; height:30px; border-radius:50%; display:flex; align-items:center;
    justify-content:center; font-size:12px; font-weight:700; flex-shrink:0;
    font-family:'Inter',sans-serif;
}
.pd-done   { background:rgba(20,184,166,0.20); color:#5ECFC4; border:1.5px solid rgba(20,184,166,0.40); }
.pd-active { background:rgba(108,99,255,0.25); color:#A09CF8; border:1.5px solid rgba(108,99,255,0.50); box-shadow:0 0 12px rgba(108,99,255,0.30); }
.pd-idle   { background:rgba(255,255,255,0.05); color:rgba(255,255,255,0.25); border:1.5px solid rgba(255,255,255,0.10); }
.ps-lbl { font-size:12px; font-weight:500; white-space:nowrap; }
.pl-done   { color:#5ECFC4; } .pl-active { color:#A09CF8; } .pl-idle { color:rgba(255,255,255,0.25); }
.ps-line { flex:1; height:1px; margin:0 8px; }
.pll-done { background:rgba(20,184,166,0.35); } .pll-idle { background:rgba(255,255,255,0.08); }

.proc {
    background: rgba(255,255,255,0.04); backdrop-filter:blur(16px);
    border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:18px 20px;
}
.pr-row { display:flex; align-items:center; gap:12px; padding:7px 0; border-bottom:1px solid rgba(255,255,255,0.04); }
.pr-row:last-child { border-bottom:none; }
.pr-ico { font-size:16px; width:24px; text-align:center; }
.pr-txt { flex:1; font-size:13px; color:rgba(255,255,255,0.75); }
.pr-b { font-size:11px; font-weight:600; padding:3px 10px; border-radius:99px; }
.bd { background:rgba(20,184,166,0.18); color:#5ECFC4; border:1px solid rgba(20,184,166,0.30); }
.ba { background:rgba(108,99,255,0.20); color:#A09CF8; border:1px solid rgba(108,99,255,0.35); animation:fp 1.4s ease-in-out infinite; }
.bi { background:rgba(255,255,255,0.05); color:rgba(255,255,255,0.25); border:1px solid rgba(255,255,255,0.08); }
@keyframes fp { 0%,100%{opacity:1}50%{opacity:0.45} }

.hw { display:flex; align-items:flex-start; gap:14px; margin-bottom:16px; }
.hw-ico { width:38px; height:38px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:18px; flex-shrink:0; backdrop-filter:blur(8px); }
.hw-t { font-size:13px; font-weight:600; color:#fff; margin-bottom:3px; }
.hw-s { font-size:12px; color:rgba(255,255,255,0.45); line-height:1.6; }

.url-strip {
    background:rgba(255,255,255,0.04); backdrop-filter:blur(16px);
    border:1px solid rgba(255,255,255,0.08); border-radius:14px;
    padding:12px 18px; display:flex; align-items:center; gap:12px; margin-bottom:18px;
}
.url-txt { font-size:12px; color:rgba(255,255,255,0.40); flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-family:'Inter',monospace; }

.sb-logo { display:flex; align-items:center; gap:12px; padding:22px 4px 0; margin-bottom:10px; }
.sb-ico { width:40px; height:40px; border-radius:12px; background:linear-gradient(135deg,#6C63FF,#14B8A6); display:flex; align-items:center; justify-content:center; font-size:20px; box-shadow:0 0 20px rgba(108,99,255,0.35); }
.sb-title { font-family:'Syne',sans-serif; font-size:17px; font-weight:700; color:#fff; }
.sb-tag { font-size:11px; color:rgba(255,255,255,0.35); margin-top:1px; }
.sb-div { height:1px; background:rgba(255,255,255,0.06); margin:16px 0; }
.sb-sec { font-size:10px; font-weight:600; color:rgba(255,255,255,0.30); letter-spacing:1.2px; text-transform:uppercase; margin-bottom:12px; }

.conn-row {
    display:flex; align-items:center; gap:10px;
    background:rgba(20,184,166,0.10); border:1px solid rgba(20,184,166,0.25);
    border-radius:12px; padding:11px 14px; margin-bottom:10px;
}
.dot-on { width:8px; height:8px; border-radius:50%; background:#14B8A6; box-shadow:0 0 8px rgba(20,184,166,0.60); animation:dotpulse 2s ease-in-out infinite; flex-shrink:0; }
@keyframes dotpulse { 0%,100%{opacity:1}50%{opacity:0.4} }
.conn-lbl { font-size:13px; font-weight:600; color:#5ECFC4; }

.feat {
    background:rgba(255,255,255,0.04); backdrop-filter:blur(20px);
    border:1px solid rgba(255,255,255,0.08); border-radius:20px;
    padding:28px 22px; text-align:center; transition:all 0.25s; position:relative; overflow:hidden;
}
.feat::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,0.14),transparent); }
.feat:hover { transform:translateY(-4px); border-color:rgba(255,255,255,0.16); }
.feat-em { font-size:32px; margin-bottom:14px; }
.feat-t { font-family:'Syne',sans-serif; font-size:15px; font-weight:700; color:#fff; margin-bottom:8px; }
.feat-s { font-size:12px; color:rgba(255,255,255,0.45); line-height:1.65; }

.hero { padding:56px 0 36px; }
.hero-tag { display:inline-flex; align-items:center; gap:8px; background:rgba(108,99,255,0.15); border:1px solid rgba(108,99,255,0.30); border-radius:99px; padding:6px 16px; font-size:12px; color:#A09CF8; font-weight:600; letter-spacing:0.5px; margin-bottom:24px; }
.hero-h { font-family:'Syne',sans-serif; font-size:48px; font-weight:800; color:#fff; line-height:1.15; margin-bottom:16px; }
.hero-h span { background:linear-gradient(135deg,#6C63FF,#14B8A6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.hero-s { font-size:16px; color:rgba(255,255,255,0.50); line-height:1.75; max-width:460px; margin-bottom:32px; }
.hero-hint { display:inline-flex; align-items:center; gap:8px; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.10); border-radius:14px; padding:14px 20px; font-size:13px; color:rgba(255,255,255,0.55); }

.sec-h { font-family:'Syne',sans-serif; font-size:22px; font-weight:700; color:#fff; margin-bottom:6px; }
.sec-s { font-size:13px; color:rgba(255,255,255,0.40); margin-bottom:22px; line-height:1.65; }
</style>
""", unsafe_allow_html=True)

for k, v in [("keys_verified", False), ("retriever", None), ("chat_history", []),
             ("video_url", None), ("chunk_count", 0), ("word_count", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v

def verify_hf_token(token):
    try:
        HfApi().whoami(token=token)
        return True
    except Exception:
        return False

def pipeline_html(active=None):
    steps = ["Transcript", "Chunking", "Embeddings", "Retriever", "Answer"]
    parts = []
    for i, lbl in enumerate(steps):
        s = "idle" if active is None else ("done" if i < active else ("active" if i == active else "idle"))
        ico = "✓" if s == "done" else ("↗" if s == "active" else str(i + 1))
        parts.append(f'<div class="ps-step"><div class="ps-dot pd-{s}">{ico}</div><div class="ps-lbl pl-{s}">{lbl}</div></div>')
        if i < len(steps) - 1:
            lc = "pll-done" if s == "done" else "pll-idle"
            parts.append(f'<div class="ps-line {lc}"></div>')
    return f'<div class="pipe">{"".join(parts)}</div>'

def proc_html(active=None):
    items = [("📥","Fetch transcript"),("✂️","Split into chunks"),("🧠","Build vector store"),("🔍","Prepare retriever")]
    rows = ""
    for i,(ico,lbl) in enumerate(items):
        if active is None: cls,txt = "bi","pending"
        elif i < active:   cls,txt = "bd","done ✓"
        elif i == active:  cls,txt = "ba","running…"
        else:              cls,txt = "bi","pending"
        rows += f'<div class="pr-row"><span class="pr-ico">{ico}</span><span class="pr-txt">{lbl}</span><span class="pr-b {cls}">{txt}</span></div>'
    return f'<div class="proc">{rows}</div>'

with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
      <div class="sb-ico">🎓</div>
      <div>
        <div class="sb-title">AskTube</div>
        <div class="sb-tag">AI Learning Assistant</div>
      </div>
    </div>
    <div class="sb-div"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">HuggingFace API Key</div>', unsafe_allow_html=True)
    st.link_button("🔑 Get your free API key", "https://huggingface.co/settings/tokens", use_container_width=True)
    hf_input = st.text_input("API Key", type="password", placeholder="hf_•••••••••••••", label_visibility="collapsed")

    if not st.session_state.keys_verified:
        if st.button("Connect →", type="primary", use_container_width=True):
            if not hf_input:
                st.warning("Paste your API key first.")
            else:
                with st.spinner("Verifying…"):
                    ok = verify_hf_token(hf_input)
                if ok:
                    st.session_state.keys_verified = True
                    st.session_state.hf_api_key = hf_input
                    st.rerun()
                else:
                    st.error("Invalid API key. Try again.")
    else:
        st.markdown('<div class="conn-row"><div class="dot-on"></div><div class="conn-lbl">HuggingFace connected</div></div>', unsafe_allow_html=True)
        if st.button("Disconnect", type="secondary", use_container_width=True):
            for k in ["keys_verified","retriever","chat_history","video_url","chunk_count","word_count"]:
                st.session_state[k] = False if k=="keys_verified" else ([] if k=="chat_history" else (None if k not in ["chunk_count","word_count"] else 0))
            st.rerun()

    st.markdown('<div class="sb-div"></div>', unsafe_allow_html=True)

    if st.session_state.video_url:
        st.markdown('<div class="sb-sec">💬 Ask the AI</div>', unsafe_allow_html=True)
        chat_box = st.container(height=350)
        with chat_box:
            if not st.session_state.chat_history:
                st.chat_message("assistant").markdown("Hey! 👋 I've read the whole video. Ask me anything.")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if user_q := st.chat_input("Ask anything about the video…"):
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with chat_box:
                st.chat_message("user").markdown(user_q)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        docs = st.session_state.retriever.invoke(user_q)
                        answer = generate_answer(user_q, docs, st.session_state.hf_api_key)
                    st.markdown(answer)
                    with st.expander("📄 Sources"):
                        for i, d in enumerate(docs, 1):
                            st.caption(f"Chunk {i}")
                            st.write(d.page_content)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        st.markdown('<div style="text-align:center;padding:24px 8px;"><div style="font-size:26px;margin-bottom:10px;">💬</div><div style="font-size:12px;color:rgba(255,255,255,0.30);line-height:1.7;">Chat unlocks after you load a video.</div></div>', unsafe_allow_html=True)

if not st.session_state.keys_verified:
    st.markdown("""
    <div class="hero">
      <div class="hero-tag">✦ AI-Powered Learning</div>
      <div class="hero-h">Learn smarter,<br>not <span>harder</span></div>
      <div class="hero-s">Paste any YouTube video and chat with it like a tutor. Instant answers from the actual content — no fluff, no guessing.</div>
      <div class="hero-hint">👈 Connect your HuggingFace key in the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    for col, em, glow, title, desc in [
        (c1,"🎬","rgba(108,99,255,0.15)","Paste any video","Works with any YouTube video that has subtitles or auto-captions."),
        (c2,"🧠","rgba(20,184,166,0.12)","AI reads it all","The entire transcript is embedded into a searchable vector database."),
        (c3,"⚡","rgba(244,114,182,0.12)","Instant answers","Get accurate answers in seconds, grounded in the video itself."),
    ]:
        with col:
            st.markdown(f'<div class="feat" style="background:linear-gradient(145deg,{glow},rgba(255,255,255,0.02));"><div class="feat-em">{em}</div><div class="feat-t">{title}</div><div class="feat-s">{desc}</div></div>', unsafe_allow_html=True)

else:
    if not st.session_state.video_url:
        st.markdown('<div class="sec-h">Load a video 🎬</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-s">Paste a YouTube link below. We\'ll fetch the transcript, chunk it, embed it, and make it searchable — in under 30 seconds.</div>', unsafe_allow_html=True)

        left, right = st.columns([3, 2], gap="large")
        with left:
            video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…", label_visibility="collapsed")
            st.caption("Open YouTube → Share → Copy link → paste above")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Process video →", type="primary"):
                if not video_url:
                    st.warning("Paste a YouTube link first.")
                else:
                    ph = st.empty()
                    try:
                        ph.markdown(proc_html(0), unsafe_allow_html=True)
                        transcript = fetch_transcript(video_url)

                        ph.markdown(proc_html(1), unsafe_allow_html=True)
                        docs = split_text(transcript)

                        ph.markdown(proc_html(2), unsafe_allow_html=True)
                        vectorstore = create_vector_store(docs, st.session_state.hf_api_key)

                        ph.markdown(proc_html(3), unsafe_allow_html=True)
                        st.session_state.retriever = get_retriever(st.session_state.hf_api_key, vectorstore, k=4)

                        st.session_state.video_url = video_url
                        st.session_state.chat_history = []
                        st.session_state.chunk_count = len(docs)
                        st.session_state.word_count = sum(len(d.page_content.split()) for d in docs)

                        ph.markdown(proc_html(4), unsafe_allow_html=True)
                        st.success("✅ Ready! Head to the sidebar to chat.")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        ph.empty()
                        st.error(f"Something went wrong: {e}")

        with right:
            st.markdown("""
            <div class="gc" style="padding:22px 24px;">
              <div style="font-family:'Syne',sans-serif;font-size:14px;font-weight:700;color:#fff;margin-bottom:18px;opacity:0.70;letter-spacing:0.5px;text-transform:uppercase;">How it works</div>
              <div class="hw"><div class="hw-ico" style="background:rgba(108,99,255,0.18);border:1px solid rgba(108,99,255,0.25);">📥</div><div><div class="hw-t">Fetch transcript</div><div class="hw-s">Full subtitles pulled directly from YouTube's API.</div></div></div>
              <div class="hw"><div class="hw-ico" style="background:rgba(20,184,166,0.18);border:1px solid rgba(20,184,166,0.25);">✂️</div><div><div class="hw-t">Chunk &amp; embed</div><div class="hw-s">Split into overlapping chunks, encoded as 768-dim vectors.</div></div></div>
              <div class="hw"><div class="hw-ico" style="background:rgba(244,114,182,0.18);border:1px solid rgba(244,114,182,0.25);">🔍</div><div><div class="hw-t">Semantic retrieval</div><div class="hw-s">Your question finds the top-4 most relevant chunks.</div></div></div>
              <div class="hw" style="margin-bottom:0;"><div class="hw-ico" style="background:rgba(251,191,36,0.15);border:1px solid rgba(251,191,36,0.25);">💬</div><div><div class="hw-t">AI answers</div><div class="hw-s">LLM responds using only what was said in the video.</div></div></div>
            </div>
            """, unsafe_allow_html=True)

    else:
        wc = f"{st.session_state.word_count:,}" if st.session_state.word_count else "—"
        cc = str(st.session_state.chunk_count) if st.session_state.chunk_count else "—"

        st.markdown(f"""
        <div class="url-strip">
          <span class="pill pv">📺 YouTube</span>
          <span class="url-txt">{st.session_state.video_url}</span>
          <span class="pill pt">✓ Indexed</span>
        </div>
        {pipeline_html(4)}
        """, unsafe_allow_html=True)

        vid_col, panel_col = st.columns([3, 2], gap="large")

        with vid_col:
            st.video(st.session_state.video_url)

            st.markdown(f"""
            <div class="stat-row">
              <div class="stat-tile">
                <div class="stat-n" style="background:linear-gradient(135deg,#6C63FF,#A09CF8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{wc}</div>
                <div class="stat-l">Transcript words</div>
              </div>
              <div class="stat-tile">
                <div class="stat-n" style="background:linear-gradient(135deg,#14B8A6,#5ECFC4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{cc}</div>
                <div class="stat-l">Text chunks</div>
              </div>
              <div class="stat-tile">
                <div class="stat-n" style="background:linear-gradient(135deg,#F472B6,#F992C8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">768</div>
                <div class="stat-l">Embed dimensions</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("↩ Load different video", type="secondary"):
                for k in ["video_url","retriever","chat_history","chunk_count","word_count"]:
                    st.session_state[k] = [] if k=="chat_history" else (None if k not in ["chunk_count","word_count"] else 0)
                st.rerun()

        with panel_col:
            tab1, tab2, tab3 = st.tabs(["💡 Insights", "📊 Retrieval", "⚙️ Pipeline"])

            with tab1:
                st.markdown("""
                <div class="ins"><div class="ins-h"><span class="pill pv">Concept</span><span class="pill py">02:14</span></div><div class="ins-b">Self-attention allows every token to attend to all others via learned query, key, and value projections — computing relevance scores across the full sequence.</div></div>
                <div class="ins"><div class="ins-h"><span class="pill pp">Key point</span><span class="pill py">07:41</span></div><div class="ins-b">Multi-head attention splits computation into parallel heads, each capturing a different type of linguistic or semantic relationship.</div></div>
                <div class="ins"><div class="ins-h"><span class="pill pt">Summary</span><span class="pill py">15:02</span></div><div class="ins-b">Positional encodings inject sequence order using sinusoidal functions — essential since the attention operation itself is permutation-invariant.</div></div>
                <div class="ins"><div class="ins-h"><span class="pill ps">Deep dive</span><span class="pill py">21:30</span></div><div class="ins-b">Feed-forward sublayers follow attention in each encoder block, expanding then contracting dimensionality with GELU activations.</div></div>
                """, unsafe_allow_html=True)

            with tab2:
                st.markdown("""
                <div class="gc" style="padding:18px 20px;">
                  <div style="font-size:11px;font-weight:600;color:rgba(255,255,255,0.30);letter-spacing:1px;text-transform:uppercase;margin-bottom:14px;">Semantic similarity · top-4</div>
                  <div class="sc-row"><div class="sc-rk">#1</div><div class="sc-bg"><div class="sc-b" style="width:93%;background:linear-gradient(90deg,#6C63FF,#A09CF8);"></div></div><div class="sc-v" style="color:#A09CF8;">0.93</div></div>
                  <div class="sc-row"><div class="sc-rk">#2</div><div class="sc-bg"><div class="sc-b" style="width:87%;background:linear-gradient(90deg,#14B8A6,#5ECFC4);"></div></div><div class="sc-v" style="color:#5ECFC4;">0.87</div></div>
                  <div class="sc-row"><div class="sc-rk">#3</div><div class="sc-bg"><div class="sc-b" style="width:74%;background:linear-gradient(90deg,#F472B6,#F992C8);"></div></div><div class="sc-v" style="color:#F992C8;">0.74</div></div>
                  <div class="sc-row"><div class="sc-rk">#4</div><div class="sc-bg"><div class="sc-b" style="width:61%;background:linear-gradient(90deg,#63B3ED,#93C5FD);"></div></div><div class="sc-v" style="color:#93C5FD;">0.61</div></div>
                </div>
                <div style="display:flex;flex-wrap:wrap;gap:7px;margin-top:4px;">
                  <span class="pill pv">top-k = 4</span>
                  <span class="pill pt">cosine similarity</span>
                  <span class="pill ps">768 dims</span>
                  <span class="pill py">FAISS index</span>
                </div>
                """, unsafe_allow_html=True)

            with tab3:
                st.markdown(proc_html(4), unsafe_allow_html=True)
                st.markdown('<div style="margin-top:14px;padding:14px 16px;background:rgba(20,184,166,0.08);border:1px solid rgba(20,184,166,0.20);border-radius:12px;font-size:13px;color:rgba(255,255,255,0.55);line-height:1.7;">All pipeline steps completed. Your video is fully indexed and ready. Use the sidebar chat to ask questions about the content.</div>', unsafe_allow_html=True)