# 📰 AI News Assistant

> A smart, context-aware news assistant that fetches the latest articles on any topic and answers your questions using semantic search and Google Gemini — all in an interactive chat interface.

---

## ✨ Features

- 🔍 **Live News Fetching** — Pulls recent articles from NewsAPI by topic and date range
- 🧹 **HTML Cleaning** — Strips raw HTML for accurate text processing via BeautifulSoup4
- 🗂️ **LangChain Integration** — Converts articles into chunked, retrieval-ready documents
- 🧠 **Semantic Search** — Google Gemini embeddings stored in a FAISS vector store
- 💬 **AI-Powered Answers** — Grounded responses generated from retrieved news context
- 📜 **Chat History** — Tracks your conversation across the session
- 🔗 **Source Transparency** — Expandable sources panel shows original articles

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Language** | Python 3.11+ |
| **Frontend** | Streamlit |
| **LLM & Embeddings** | LangChain + Google Gemini (GenAI) |
| **Vector Store** | FAISS |
| **News Source** | NewsAPI |
| **HTML Parsing** | BeautifulSoup4, Requests |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

### 2. Install dependencies

```bash
pip install requests beautifulsoup4 streamlit \
  langchain-core langchain-classic langchain-community \
  langchain-google-genai faiss-cpu
```

### 3. Obtain API keys

| Key | Link |
|---|---|
| NewsAPI Key | https://newsapi.org/register |
| Google Gemini API Key | https://aistudio.google.com/app/apikey |

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🚀 Usage

1. Open the app in your browser (default: `http://localhost:8501`)
2. Enter your **NewsAPI Key** and **Google Gemini Key** in the sidebar
3. Click **Verify Keys** to authenticate
4. Select a **topic**, **date range**, and **number of document chunks** to retrieve
5. Click **Fetch & Process** to load and index the latest articles
6. Ask questions in the chat — expand **Sources** to see the original articles behind each answer

---

## 📁 Project Structure

```
project/
├── app.py                   # Streamlit frontend and chat interface
├── news_fetcher.py          # Fetch and filter news from NewsAPI
├── document_processor.py   # Convert articles to LangChain documents
├── vector_store.py          # Chunk documents and build FAISS index
└── llm_chain.py             # Retrieve context and generate answers via Gemini
```

---

## 💡 Notes

- Answers are grounded in the fetched news context. If relevant articles weren't retrieved, the model may fall back to general knowledge or indicate insufficient information.
- For best results, use a specific topic (e.g., `"AI regulation"` rather than `"tech"`).
- Python 3.11+ is recommended for full compatibility with all dependencies.

---
