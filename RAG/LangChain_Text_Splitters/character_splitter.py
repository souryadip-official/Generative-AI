from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
loader = TextLoader(file_path='cricket.txt')
documents = loader.load()
splitter = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=10
)
chunks = splitter.split_documents(documents=documents)
for chunk in chunks:
    print(chunk.page_content, '\n')