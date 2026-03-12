from langchain_community.retrievers import WikipediaRetriever
retriever = WikipediaRetriever(
    top_k_results=2, # number of similar documents we want to fetch
    lang='en', # language
)

user_query = input('Enter any query: ')
docs = retriever.invoke(input=user_query) # Returns a list of objects of Document class
for idx, doc in enumerate(docs):
    print(f'\n*-------- Result {idx+1} --------*\nContent: {doc.page_content}\n')
print('-'*10, 'END', '-'*10)