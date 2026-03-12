import requests as req
from datetime import datetime, timedelta

def fetch_news(news_api_key:str, query: str, days:int=2, page_size:int=20):
    url = "https://newsapi.org/v2/everything"
    from_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "from": from_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": news_api_key
    }
    response = req.get(url, params=params)
    if response.status_code != 200:
        raise Exception("NEWS API request failed!")
    else:
        data = response.json()
        return data['articles']