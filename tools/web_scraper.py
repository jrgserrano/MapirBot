import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
import re

@tool
def scrape_web(url: str) -> str:
    """Scrape the content of a web page and return the text."""
    try:
        # Add a headers to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing whitespace
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit the output length to avoid token explosion
        # If it's too long, just take the first 4000 characters
        if len(text) > 4000:
            text = text[:4000] + "\n... (omitted)"
            
        return f"Content from {url}:\n\n{text}"
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"
