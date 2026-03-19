import asyncio
from openai import AsyncOpenAI
import json

async def main():
    client = AsyncOpenAI(
        base_url="http://127.0.0.1:1234/v1",
        api_key="lm-studio"
    )
    
    print("Testing chat completion...")
    try:
        response = await client.chat.completions.create(
            model="google/gemma-3-4b",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("Chat response:", response.choices[0].message.content)
    except Exception as e:
        print("Chat Error:", e)

    print("\nTesting embeddings...")
    try:
        response = await client.embeddings.create(
            model="text-embedding-nomic-embed-text-v1.5",
            input=["This is a test of the embedding system."]
        )
        print("Embedding length:", len(response.data[0].embedding))
    except Exception as e:
        print("Embedding Error:", e)

if __name__ == "__main__":
    asyncio.run(main())
