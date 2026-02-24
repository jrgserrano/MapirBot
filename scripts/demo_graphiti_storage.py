import asyncio
import os
from datetime import datetime, timezone
import uuid
from dotenv import load_dotenv

# Import Graphiti components
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.nodes import EpisodeType, EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

# Load environment variables
load_dotenv()

async def main():
    # 1. Configuration
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'neo4j_changed')

    # LLM Configuration (using Ollama as seen in test_graphiti.py)
    llm_name = os.environ.get('MODEL_NAME', 'llama3.2:3b')
    openai_base_url = os.environ.get('OPENAI_BASE_URL', 'http://localhost:11434/v1')
    
    llm_config = LLMConfig(
        api_key='ollama',
        base_url=openai_base_url,
        model=llm_name,
        small_model=llm_name
    )
    llm_client = OpenAIClient(config=llm_config)

    # Embedder Configuration
    embedder_config = OpenAIEmbedderConfig(
        api_key='ollama',
        base_url=openai_base_url,
        embedding_model='mxbai-embed-large'
    )
    embedder = OpenAIEmbedder(config=embedder_config)

    # Initialize Graphiti
    client = Graphiti(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        llm_client=llm_client,
        embedder=embedder
    )

    print("[INFO] Clearing existing data...")
    await clear_data(client.driver)
    
    print("[INFO] Building indices...")
    await client.build_indices_and_constraints()

    # --- METHOD 1: add_episode (Semantic Storage) ---
    # This is the easiest way. You give it text, and Graphiti uses the LLM 
    # to automatically extract nodes (entities) and edges (facts).
    
    print("\n--- METHOD 1: add_episode ---")
    print("Storing: 'Alice works as a robotics engineer at Mapir Labs in Malaga.'")
    
    await client.add_episode(
        name='Staff Info',
        episode_body='Alice works as a robotics engineer at Mapir Labs in Malaga.',
        source_description='Manual Entry',
        source=EpisodeType.text,
        reference_time=datetime.now(timezone.utc),
    )

    # --- METHOD 2: add_triplet (Relational Storage) ---
    # Use this if you already have structured data and don't want to use an LLM
    # to extract it. It allows for precise control.
    
    print("\n--- METHOD 2: add_triplet ---")
    print("Storing: Alice -> LIKES -> Coffee")

    # Define nodes
    source_node = EntityNode(
        name="Alice",
        labels=["Person"],
        group_id="Default",
        created_at=datetime.now(timezone.utc)
    )
    
    target_node = EntityNode(
        name="Coffee",
        labels=["Object"],
        group_id="Default",
        created_at=datetime.now(timezone.utc)
    )

    # Define edge (relation)
    edge = EntityEdge(
        source_node_uuid=source_node.uuid, # These will be resolved during add_triplet
        target_node_uuid=target_node.uuid,
        name="LIKES",
        fact="Alice likes coffee",
        relation="LIKES",
        group_id="Default",
        created_at=datetime.now(timezone.utc)
    )

    # Add the triplet
    await client.add_triplet(source_node, edge, target_node)
    print("[INFO] Triplet added.")

    # --- RETRIEVAL: search ---
    print("\n--- SEARCHING THE GRAPH ---")
    
    query = "What do we know about Alice?"
    print(f"Query: {query}")
    
    results = await client.search(query)
    
    print("\nFound Facts:")
    for edge in results:
        print(f"- {edge.fact}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
