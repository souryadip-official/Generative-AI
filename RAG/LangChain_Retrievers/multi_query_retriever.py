import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.multi_query import MultiQueryRetriever # The langchain-classic package is a separate library that contains legacy functionality from LangChain v0.x, which has been moved out of the main langchain package as part of the v1.0 release. It exists primarily to provide backward compatibility for existing applications
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

doc1 = Document(
    page_content="Regular exercise is essential for maintaining good health. Activities such as running, cycling, and strength training improve cardiovascular health and build muscle strength.",
    metadata={"topic": "Exercise", "category": "Fitness"}
)

doc2 = Document(
    page_content="A balanced diet that includes proteins, carbohydrates, healthy fats, vitamins, and minerals helps the body maintain energy levels and supports overall health.",
    metadata={"topic": "Nutrition", "category": "Fitness"}
)

doc3 = Document(
    page_content="Yoga improves flexibility, balance, and mental relaxation. Many people practice yoga daily to reduce stress and maintain physical fitness.",
    metadata={"topic": "Yoga", "category": "Fitness"}
)

doc4 = Document(
    page_content="High-intensity interval training (HIIT) involves short bursts of intense exercise followed by rest. It is very effective for burning calories and improving stamina.",
    metadata={"topic": "HIIT", "category": "Fitness"}
)

doc5 = Document(
    page_content="Cricket is one of the most popular sports in countries like India, Australia, and England. The game is played between two teams of eleven players using a bat and ball.",
    metadata={"topic": "Cricket", "category": "Sports"}
)

doc6 = Document(
    page_content="The solar system consists of the Sun and the objects that orbit it, including eight planets, moons, asteroids, and comets. Earth is the third planet from the Sun.",
    metadata={"topic": "Solar System", "category": "Space"}
)

doc7 = Document(
    page_content="Air pollution is caused by harmful substances such as smoke, chemicals, and dust entering the atmosphere. It can cause serious health and environmental problems.",
    metadata={"topic": "Pollution", "category": "Environment"}
)

doc8 = Document(
    page_content="A healthy lifestyle includes regular exercise, good nutrition, proper sleep, and stress management. These habits help people live longer and healthier lives.",
    metadata={"topic": "Lifestyle", "category": "Health"}
)

doc9 = Document(
    page_content="Dogs are one of the most popular pets in the world. They are known for their loyalty, intelligence, and companionship with humans.",
    metadata={"topic": "Dogs", "category": "Pets"}
)

doc10 = Document(
    page_content="Cats are independent and curious animals that make great pets. Many people keep cats because they require less maintenance than some other pets.",
    metadata={"topic": "Cats", "category": "Pets"}
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

similarity_retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={"k": 3},
)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": 3},
    ),
    llm=ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
    )
)

# Fetch documents
user_query = 'How can someone stay healthy and fit in daily life and be maintain themselves?'
similarity_results = similarity_retriever.invoke(input=user_query)
multiquery_results = multiquery_retriever.invoke(input=user_query)

print('='*10, '\nNormal retriever results\n', '='*10)
for idx, doc in enumerate(similarity_results):
    print(f'\n*-------- Result {idx+1} --------*\nContent: {doc.page_content}\n')
print('-'*10, 'END', '-'*10)

print('='*10, '\nMulti-query retriever results\n', '='*10)
for idx, doc in enumerate(multiquery_results):
    print(f'\n*-------- Result {idx+1} --------*\nContent: {doc.page_content}\n')
print('-'*10, 'END', '-'*10)