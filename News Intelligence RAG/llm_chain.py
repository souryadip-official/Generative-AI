from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_answer(query: str, retrived_docs, google_api_key: str):
    context_text = ""
    # combining the retrieved documents
    for i, doc in enumerate(retrived_docs):
        context_text += f"--- Article {i} ---\n"
        context_text += doc.page_content + "\n\n"
    
    template = PromptTemplate(
        template = """
        You are an intelligent and helpful AI news assistant. 
        Your primary task is to answer the user's question using the provided NEWS CONTEXT.
        
        Guidelines:
        1. If the NEWS CONTEXT contains the answer, use it to provide a clear and concise response.
        2. If the answer is NOT in the NEWS CONTEXT, but the question is about a highly popular, universally known topic that you can answer with absolute confidence, you may answer it. However, you MUST explicitly state that you are answering based on general knowledge and not the recent news context.
        3. If the answer is not in the context and it is not a widely known fact, simply reply: "I don't have enough information in the provided news to answer that."
        
        NEWS CONTEXT: {context}
        
        USER QUESTION: {question}
        
        ANSWER:""",
        input_variables=['context', 'question']
    )
    
    model = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        google_api_key=google_api_key,
        temperature=0.3
    )
    
    parser = StrOutputParser()
    
    chain = template | model | parser
    response = chain.invoke({
        'context': context_text,
        'question': query
    })
    
    return response