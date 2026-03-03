from ddgs import DDGS
from langchain_core.tools import tool
from typing import List, Dict

@tool
def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search the web using DuckDuckGo and return a list of relevant links and snippets.
    Use this to find URLs for deeper scraping.
    """
    try:
        results = []
        with DDGS() as ddgs:
            ddgs_gen = ddgs.text(query, max_results=max_results)
            for r in ddgs_gen:
                results.append({
                    "title": r.get("title"),
                    "link": r.get("href"),
                    "snippet": r.get("body")
                })
        return results
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]
