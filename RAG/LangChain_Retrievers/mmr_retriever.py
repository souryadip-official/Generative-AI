import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS # Facebook AI Similarity Search (Another vector store like ChromaDB)

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

doc1 = Document(
    page_content="Sachin Tendulkar is one of the greatest cricketers in history. He played for India from 1989 to 2013 and is known as the 'God of Cricket'. He holds the record for the most runs in international cricket and was the first player to score 100 international centuries.",
    metadata={"player": "Sachin Tendulkar", "role": "Batsman", "team": "MI"}
)

doc2 = Document(
    page_content="Virat Kohli is one of the most successful modern cricketers and a former captain of the Indian cricket team. He is known for his aggressive batting style and consistency across formats. Kohli has scored thousands of international runs and numerous centuries.",
    metadata={"player": "Virat Kohli", "role": "Batsman", "team": "RCB"}
)

doc3 = Document(
    page_content="MS Dhoni is one of the most successful captains in cricket history. Under his leadership India won the 2007 T20 World Cup, the 2011 ODI World Cup, and the 2013 Champions Trophy. He is famous for his calm leadership and finishing ability.",
    metadata={"player": "MS Dhoni", "role": "Wicketkeeper Batsman", "team": "CSK"}
)

doc4 = Document(
    page_content="Rohit Sharma is the captain of the Indian cricket team in limited overs formats and is known for his elegant batting. He holds the record for the highest individual score in ODI cricket, scoring 264 runs in a single match.",
    metadata={"player": "Rohit Sharma", "role": "Opening Batsman", "team": "MI"}
)

doc5 = Document(
    page_content="Kapil Dev is one of India's greatest all-rounders and captained India to its first World Cup victory in 1983. He was known for his fast bowling and powerful batting and was one of the best all-rounders of his time.",
    metadata={"player": "Kapil Dev", "role": "All Rounder", "team": "N/A"}
)

doc6 = Document(
    page_content="Rahul Dravid, known as 'The Wall', was one of the most dependable batsmen in cricket history. He was famous for his solid technique and ability to play long innings. Dravid scored over 13,000 runs in Test cricket.",
    metadata={"player": "Rahul Dravid", "role": "Batsman", "team": "RR"}
)

doc7 = Document(
    page_content="Anil Kumble is one of India's greatest spin bowlers and took more than 600 wickets in Test cricket. He once took all 10 wickets in a single Test innings against Pakistan, becoming only the second bowler in history to achieve this feat.",
    metadata={"player": "Anil Kumble", "role": "Spin Bowler", "team": "RCB"}
)

doc8 = Document(
    page_content="Jasprit Bumrah is one of the best fast bowlers in modern cricket. Known for his unique bowling action and deadly yorkers, he has been a key player for India in all formats of the game.",
    metadata={"player": "Jasprit Bumrah", "role": "Fast Bowler", "team": "MI"}
)

doc9 = Document(
    page_content="Ravindra Jadeja is one of the top all-rounders in international cricket. He is known for his accurate left-arm spin bowling, aggressive batting, and exceptional fielding skills.",
    metadata={"player": "Ravindra Jadeja", "role": "All Rounder", "team": "RR"}
)

doc10 = Document(
    page_content="Sunil Gavaskar was one of the greatest opening batsmen in cricket history. He was the first player to score 10,000 runs in Test cricket and was known for his excellent technique against fast bowling.",
    metadata={"player": "Sunil Gavaskar", "role": "Opening Batsman", "team": "N/A"}
)
documents = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9, doc10]

vector_store = FAISS.from_documents(
    embedding=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=google_api_key,
    ),
    documents=documents,
)

# Converting the vector store into a retriever
retriever = vector_store.as_retriever(
    search_type='mmr', # this enables MMR
    search_kwargs={"k": 3, "lambda_mult": 0.6}, # k = top results, lambda_mult = relevance-diversity balance (within range of [0,1]. If it is 0, we get very diverse results)
)

user_query = 'Who among these is the most successful captain?'
docs = retriever.invoke(input=user_query)

for idx, doc in enumerate(docs):
    print(f'\n*-------- Result {idx+1} --------*\nContent: {doc.page_content}\n')
print('-'*10, 'END', '-'*10)