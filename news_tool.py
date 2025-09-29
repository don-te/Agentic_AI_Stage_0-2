import os
from dotenv import load_dotenv
from newsapi import NewsApiClient


load_dotenv()  # Load environment variables from .env file

def fetch_news(query: str):
    """Fetches the latest news articles for a given query."""
    # This API key should be stored in your .env file
    api_key = os.environ.get("NEWSAPI_KEY") 

    # Init with your API key
    newsapi = NewsApiClient(api_key=api_key)

    # /v2/everything endpoint for all articles
    all_articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=5)

    # Extract the content from the first 3 articles
    articles = ""
    for article in all_articles['articles'][:3]:
        articles += f"Title: {article['title']}\n"
        articles += f"Description: {article['description']}\n"
        articles += f"Content: {article['content']}\n\n"

    return articles