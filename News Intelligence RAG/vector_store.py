from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200, # for the context
    )
    return text_splitter.split_documents(docs)

def create_vector_store(splits, google_api_key:str):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=google_api_key,
    )
    vector_store = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )
    return vector_store

def get_retriever(vector_store: FAISS, k:int=3):
    retriever = vector_store.as_retriever(
        search_type = 'mmr', # maximal marginal relevance
        search_kwargs={"k": k}
    )
    return retriever