from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
loader = TextLoader(file_path='cricket.txt')
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)
chunks = splitter.split_documents(documents=documents)
for chunk in chunks:
    print(chunk.page_content, '\n')