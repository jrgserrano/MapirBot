import requests
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_get_graph():
    url = "https://sagem-aluminium-mud-heading.trycloudflare.com/get_graph"
    headers = {
        "Content-Type": "application/json"
    }

    try:
        print(f"Sending request to {url}...")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            print("Success!")
            print("Response:", json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.status_code}")
            print("Detail:", response.text)
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_get_graph()