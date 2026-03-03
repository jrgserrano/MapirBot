import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.web_scraper import scrape_web

def test_scraper():
    # Test with a URL that often requires headers (e.g., a news site or github)
    test_urls = [
        "https://www.google.com",
        "https://github.com/hrithikkoduri/WebRover"
    ]
    
    for url in test_urls:
        print(f"Testing scraper for: {url}")
        result = scrape_web.invoke(url)
        if "Error scraping" in result:
            print(f"FAILED: {result}\n")
        else:
            print(f"SUCCESS: {len(result)} characters retrieved.\n")

if __name__ == "__main__":
    test_scraper()
