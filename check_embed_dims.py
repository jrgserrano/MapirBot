import httpx
import json

def check_embeddings():
    url = "http://localhost:11434/v1/embeddings"
    payload = {
        "model": "mxbai-embed-large",
        "input": "This is a test"
    }
    try:
        response = httpx.post(url, json=payload, timeout=10)
        data = response.json()
        embedding = data['data'][0]['embedding']
        print(f"Dimension: {len(embedding)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_embeddings()
