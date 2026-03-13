from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

def split_text(transcript: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    docs = splitter.create_documents([transcript])
    return docs

def create_vector_store(docs, huggingfacehub_api_token):
    embeddings = HuggingFaceEndpointEmbeddings(
        model = "intfloat/multilingual-e5-large",
        huggingfacehub_api_token = huggingfacehub_api_token
    )
    vector_store = FAISS.from_documents(
        documents = docs,
        embedding = embeddings
    )
    return vector_store

def get_retriever(huggingfacehub_api_token, vector_store: FAISS, k: int = 5):
    compression_retriever = ContextualCompressionRetriever(
        base_retriever = vector_store.as_retriever(
            search_type = "mmr",
            search_kwargs = {"k": k},
        ),
        base_compressor = LLMChainExtractor.from_llm(
            ChatHuggingFace(
                llm = HuggingFaceEndpoint(
                    repo_id = "google/gemma-3-27b-it",
                    huggingfacehub_api_token = huggingfacehub_api_token
                )
            )
        ),
    )
    return compression_retriever