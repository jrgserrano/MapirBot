import os
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient, LLMConfig
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
from dotenv import load_dotenv

load_dotenv()

class GraphitiClient:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls._create_client()
        return cls._instance

    @staticmethod
    def _create_client():
        neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
        neo4j_password = os.environ.get('NEO4J_PASSWORD', 'neo4j_changed')

        llm_name = os.environ.get('MODEL_NAME', 'llama3.2:3b')
        openai_base_url = os.environ.get('OPENAI_BASE_URL', 'http://localhost:11434/v1')
        
        # LLM Config
        llm_config = LLMConfig(
            api_key='ollama',
            base_url=openai_base_url,
            model=llm_name,
            small_model=llm_name
        )
        llm_client = OpenAIClient(config=llm_config)

        # Embedder Config
        embedder_config = OpenAIEmbedderConfig(
            api_key='ollama',
            base_url=openai_base_url,
            embedding_model='mxbai-embed-large'
        )
        embedder = OpenAIEmbedder(config=embedder_config)

        # Initialize Graphiti
        return Graphiti(
            neo4j_uri,
            neo4j_user,
            neo4j_password,
            llm_client=llm_client,
            embedder=embedder
        )

# Global access point
graphiti_client = None

def get_graphiti():
    global graphiti_client
    if graphiti_client is None:
        graphiti_client = GraphitiClient.get_instance()
    return graphiti_client
