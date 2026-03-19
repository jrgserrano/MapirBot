"""Mapir Memory MCP Server.

Provides a robust, synchronized memory system for MapirBot using Graphiti and Neo4j.
Optimized for local LLMs and reliable data persistence.
"""

import os
import yaml
import logging
import sys
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

from services.memory_service import MapirMemoryService

# --- Configuration ---

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config_data = load_config()

# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("mapir-memory")

# --- Service Instance ---

memory_service = MapirMemoryService(config_data)

# --- MCP Server ---

INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph. 
It captures relationships between concepts, entities, and information as episodes, nodes, and facts.

Key capabilities:
1. Add memories (text, message, or JSON) with add_memory. Processing is queued sequentially.
2. Search for nodes (entities) using search_nodes.
3. Search for facts (relationships) using search_facts.
4. Search for episodes (raw context) using search_episodes.
5. Use get_status to monitor the background processing queue.

Each memory is organized by 'group_id' (e.g., 'user_123').
"""

mcp = FastMCP(
    "Mapir Memory",
    instructions=INSTRUCTIONS
)

# --- Tools ---

@mcp.tool()
async def add_memory(
    name: str,
    content: str,
    group_id: str,
    source: str = "text",
    source_description: str = ""
) -> str:
    """Add an episode to memory. 
    Processing is queued sequentially per group_id.
    
    Args:
        name: Name of the episode.
        content: The memory content (text, JSON string, or transcript).
        group_id: Unique identifier for the context (e.g. user ID).
        source: Type of content ('text', 'json', 'message'). Default 'text'.
        source_description: Brief description of where this came from.
    """
    try:
        qsize = await memory_service.add_episode_queued(
            name=name,
            content=content,
            group_id=group_id,
            source=source,
            source_description=source_description
        )
        return f"Memory '{name}' queued for processing. Current queue length: {qsize}"
    except Exception as e:
        logger.error(f"Failed to queue memory: {e}")
        return f"Error: {e}"

@mcp.tool()
async def search_nodes(query: str, group_id: str, limit: int = 10) -> list[dict]:
    """Search for entities (nodes) in the knowledge graph.
    
    Args:
        query: Semantic search query.
        group_id: The context to search within.
        limit: Max results to return.
    """
    try:
        nodes = await memory_service.search_nodes(query, [group_id], limit)
        return [n.model_dump(mode="json", exclude={"name_embedding"}) for n in nodes]
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

@mcp.tool()
async def search_facts(query: str, group_id: str, limit: int = 10) -> list[dict]:
    """Search for relationships (facts) in the knowledge graph.
    
    Args:
        query: Semantic search query.
        group_id: The context to search within.
        limit: Max results to return.
    """
    try:
        edges = await memory_service.search_facts(query, [group_id], limit)
        return [e.model_dump(mode="json", exclude={"fact_embedding"}) for e in edges]
    except Exception as e:
        logger.error(f"Fact search failed: {e}")
        return []

@mcp.tool()
async def search_episodes(query: str, group_id: str, limit: int = 10) -> list[dict]:
    """Search for raw episodic context in the knowledge graph.
    
    Args:
        query: Semantic search query.
        group_id: The context to search within.
        limit: Max results to return.
    """
    try:
        episodes = await memory_service.search_episodes(query, [group_id], limit)
        return [e.model_dump(mode="json") for e in episodes]
    except Exception as e:
        logger.error(f"Episode search failed: {e}")
        return []

@mcp.tool()
async def get_status(group_id: str) -> dict:
    """Get the status of the memory processing queue.
    
    Args:
        group_id: The context to check.
    """
    queue = memory_service.episode_queues.get(group_id)
    return {
        "group_id": group_id,
        "queue_length": queue.qsize() if queue else 0,
        "worker_active": memory_service.queue_workers.get(group_id, False)
    }

@mcp.tool()
async def merge_nodes(group_id: str, primary_node_name: str, duplicate_node_name: str) -> str:
    """Merge two nodes in the graph. All relationships from the duplicate will be moved to the primary.
    
    Args:
        group_id: The context to operate in.
        primary_node_name: The name of the node to keep.
        duplicate_node_name: The name of the node to merge (will be deleted).
    """
    try:
        await memory_service.merge_nodes_by_name(group_id, primary_node_name, duplicate_node_name)
        return f"Successfully merged '{duplicate_node_name}' into '{primary_node_name}'."
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        return f"Merge failed: {e}"

@mcp.tool()
async def update_node(group_id: str, name: str, new_summary: Optional[str] = None, attributes: Optional[dict] = None) -> str:
    """Update an existing node's summary or attributes. 
    Use this to correct info or change status (e.g., mark something as 'Completed' or 'Cancelled').
    
    Args:
        group_id: The context.
        name: Precise name of the node to update.
        new_summary: New summary text (replaces old one).
        attributes: Dictionary of attributes to merge/update.
    """
    try:
        await memory_service.update_node(group_id, name, new_summary, attributes)
        return f"Node '{name}' updated successfully."
    except Exception as e:
        logger.error(f"Update failed: {e}")
        return f"Error: {e}"

@mcp.tool()
async def invalidate_edge(group_id: str, edge_uuid: str) -> str:
    """Mark a specific relationship (edge) as no longer valid.
    Use this when a previously stated fact is retracted or cancelled.
    
    Args:
        group_id: The context.
        edge_uuid: The UUID of the edge to invalidate.
    """
    try:
        await memory_service.invalidate_edge(group_id, edge_uuid)
        return f"Edge '{edge_uuid}' invalidated successfully."
    except Exception as e:
        logger.error(f"Invalidate failed: {e}")
        return f"Error: {e}"

@mcp.tool()
async def clear_graph(group_id: str) -> str:
    """Clear data for a specific group_id. Use 'ALL' to wipe the entire database."""
    try:
        await memory_service.clear_graph(group_id)
        if group_id.upper() == "ALL":
            return "Entire database has been cleared."
        return f"Database cleared for group: {group_id}"
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    mcp.run()
