import asyncio
import os
import sys
from pathlib import Path

# Add mcp_server/src to path for imports
sys.path.insert(0, str(Path("mcp_servers/graphiti/mcp_server/src").absolute()))

from config.schema import GraphitiConfig
from graphiti_mcp_server import GraphitiService
from graphiti_core.nodes import EpisodeType
import logging

logging.basicConfig(level=logging.INFO)

async def main():
    # Set up env variables that LM studio would pass
    os.environ["OPENAI_API_KEY"] = "lm-studio"
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:1234/v1"
    os.environ["MODEL_NAME"] = "google/gemma-3-4b"
    os.environ["EMBEDDER_MODEL"] = "text-embedding-nomic-embed-text-v1.5"
    os.environ["DATABASE_PROVIDER"] = "neo4j"
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "demodemo"

    # Use default yaml
    os.environ["CONFIG_PATH"] = str(Path("mcp_servers/graphiti/mcp_server/config/config.yaml").absolute())
    
    config = GraphitiConfig()
    service = GraphitiService(config)
    await service.initialize()
    
    client = await service.get_client()
    
    print("Testing episode add...")
    try:
        await client.add_episode(
            name="Jorge's Info",
            episode_body="Hola, soy Jorge y me gusta el tenis.",
            source_description="text",
            source=EpisodeType.text,
            group_id="main"
        )
        print("Finished episode add.")
        
        episodes = await client.search_episodes(query="Jorge", group_ids=["main"])
        print("Search results:", episodes)
    except Exception as e:
        print("Error during add_episode:", e)

if __name__ == "__main__":
    asyncio.run(main())
