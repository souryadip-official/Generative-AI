from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(file_path='doc.pdf')
documents = loader.load() # python list of documents
document = documents[0] # <class 'langchain_core.documents.base.Document'>
content = document.page_content # Content of the first page
print(content)