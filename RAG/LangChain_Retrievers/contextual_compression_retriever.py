import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

doc1 = Document(
    page_content="The Maurya Empire was one of the largest empires in ancient India and was founded by Chandragupta Maurya. Meanwhile, climate change is causing global temperatures to rise, leading to melting glaciers and extreme weather events.",
    metadata={"id": 1}
)

doc2 = Document(
    page_content="Cricket is a popular sport played between two teams of eleven players, especially in countries like India, England, and Australia. Deforestation occurs when forests are cleared for agriculture, urban development, or logging.",
    metadata={"id": 2}
)

doc3 = Document(
    page_content="Veterinary doctors specialize in treating animals and ensuring their health and well-being. The Mughal Empire ruled large parts of India for centuries and contributed significantly to Indian architecture and culture.",
    metadata={"id": 3}
)

doc4 = Document(
    page_content="Climate change leads to rising sea levels and increased frequency of natural disasters. Cricket legends like Sachin Tendulkar have greatly influenced the popularity of the sport in India.",
    metadata={"id": 4}
)

doc5 = Document(
    page_content="Deforestation reduces biodiversity and disrupts ecosystems that many species depend on. Veterinary doctors often perform surgeries, vaccinations, and medical treatments for pets and livestock.",
    metadata={"id": 5}
)

doc6 = Document(
    page_content="The Indian independence movement involved many leaders such as Mahatma Gandhi who advocated nonviolent resistance. Climate change also affects agriculture by altering rainfall patterns and increasing drought risks.",
    metadata={"id": 6}
)

doc7 = Document(
    page_content="Cricket tournaments such as the World Cup attract millions of viewers globally. Deforestation also contributes to climate change because trees that absorb carbon dioxide are removed.",
    metadata={"id": 7}
)

doc8 = Document(
    page_content="Veterinary medicine includes diagnosing diseases in animals and ensuring food safety in livestock farming. Ancient Indian civilizations such as the Indus Valley Civilization had advanced urban planning.",
    metadata={"id": 8}
)

doc9 = Document(
    page_content="Climate change is one of the most pressing global issues today, impacting ecosystems, weather patterns, and human health. Cricket requires skills like batting, bowling, and fielding.",
    metadata={"id": 9}
)

doc10 = Document(
    page_content="Deforestation can lead to soil erosion and loss of wildlife habitats. During British rule in India, several economic and political changes shaped the country's modern history.",
    metadata={"id": 10}
)
documents = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9, doc10]

vector_store = FAISS.from_documents(
    embedding=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=google_api_key,
    ),
    documents=documents,
)

# Retrievers
compression_retriever = ContextualCompressionRetriever(
    base_retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={"k": 3},
    ),
    base_compressor = LLMChainExtractor.from_llm(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
        )
    ),
)

# Fetch documents
user_query = 'What are the environmental effects of deforestation?'
compression_retriever_results = compression_retriever.invoke(input=user_query)

print('='*10, 'Contextual Compression Retriever results', '='*10)
for idx, doc in enumerate(compression_retriever_results):
    print(f'\n*-------- Result {idx+1} --------*\nContent: {doc.page_content}\n')
print('-'*10, 'END', '-'*10)