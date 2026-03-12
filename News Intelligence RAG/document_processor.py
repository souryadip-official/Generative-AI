from bs4 import BeautifulSoup # to handle the content section (as it is HTML)
from langchain_core.documents import Document
import re

def clean_html(text):
    if not text:
        return ""   
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"\[\+\d+\s*chars\]", "", text)
    text = " ".join(text.split())
    return text

def convert_to_documents(articles):
    docs = []
    for article in articles:
        title = article.get("title", "")
        description = clean_html(article.get("description", ""))
        content = clean_html(article.get("content", ""))
        source = article["source"]["name"]
        date = article.get("publishedAt", "")
        
        article_text = f"Title: {title}\nDescription: {description}\nContent: {content}\nSource: {source}\nPublished: {date}"
        doc = Document(
            page_content = article_text,
            metadata = {
                "source": source,
                "url": article["url"]
            },
        )
        docs.append(doc)
    return docs