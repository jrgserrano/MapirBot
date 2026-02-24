from agent.graph import graph_app
from langchain_core.messages import HumanMessage
import time

def test_web_scraper():
    print("\n--- Testing Web Scraper Routing & Execution ---")
    url = "https://en.wikipedia.org/wiki/Robotics"
    query = f"Can you tell me what the first paragraph of {url} says?"
    inputs = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": "web_scraper_test"}}
    
    start_time = time.time()
    for output in graph_app.stream(inputs, config=config):
        for key, value in output.items():
            print(f"Node: {key}")
            if key == "web_scraper":
                print(f"  Scraper Executed for URL")
            if key == "final_answer":
                if "messages" in value:
                    print(f"  Final Answer: {value['messages'][-1].content[:200]}...")
    
    print(f"\nTotal Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    test_web_scraper()
