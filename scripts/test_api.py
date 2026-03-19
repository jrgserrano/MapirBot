import requests
import json

def test_ask():
    url = "https://alt-gone-ent-pensions.trycloudflare.com/ask"
    payload = {
        "query": "qué tiempo hace en málaga?",
        "thread_id": "test_thread_123"
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        
        if response.status_code == 200:
            print("Success!")
            print("Response:", json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.status_code}")
            print("Detail:", response.text)
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_ask()
