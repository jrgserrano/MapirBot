import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv

load_dotenv()

from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'neo4j')

# Configure Graphiti to use Ollama
llm_config = LLMConfig(
    api_key=os.environ.get('OPENAI_API_KEY', 'ollama'),
    base_url=os.environ.get('OPENAI_BASE_URL', 'http://localhost:11434/v1'),
    model=os.environ.get('MODEL_NAME', 'llama3.2')
)
llm_client = OpenAIClient(config=llm_config)

client = Graphiti(
    neo4j_uri,
    neo4j_user,
    neo4j_password,
    llm_client=llm_client
)

async def ingest_products_data(client: Graphiti):
    script_dir = Path.cwd().parent
    json_file_path = script_dir / 'data' / 'manybirds_products.json'

    with open(json_file_path) as file:
        products = json.load(file)['products']

    for i, product in enumerate(products):
        await client.add_episode(
            name=product.get('title', f'Product {i}'),
            episode_body=str({k: v for k, v in product.items() if k != 'images'}),
            source_description='ManyBirds products',
            source=EpisodeType.json,
            reference_time=datetime.now(timezone.utc),
        )

from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_EPISODE_MENTIONS

user_name = 'jess'

def edges_to_facts_string(entities: list[EntityEdge]):
    return '-' + '\n- '.join([edge.fact for edge in entities])

async def main():
    await clear_data(client.driver)
    print("[INFO] Data cleared")
    await client.build_indices_and_constraints()
    print("[INFO] Indices and constraints built")
    await ingest_products_data(client)
    print("[INFO] Products data ingested")

    await client.add_episode(
        name='User Creation',
        episode_body=(f'{user_name} is interested in buying a pair of shoes'),
        source=EpisodeType.text,
        reference_time=datetime.now(timezone.utc),
        source_description='SalesBot',
    )
    
    nl = await client._search(user_name, NODE_HYBRID_SEARCH_EPISODE_MENTIONS)

    user_node_uuid = nl.nodes[0].uuid

    nl = await client._search('ManyBirds', NODE_HYBRID_SEARCH_EPISODE_MENTIONS)
    manybirds_node_uuid = nl.nodes[0].uuid




if __name__ == "__main__":
    asyncio.run(main())
