import asyncio
from langchain_ollama import OllamaEmbeddings

async def run():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    vector1 = await embeddings.aembed_query("Hello world")
    vector2 = await embeddings.aembed_query("Hi there")
    
    # Cosine similarity
    import numpy as np
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"Similarity: {similarity}")

if __name__ == "__main__":
    asyncio.run(run())
