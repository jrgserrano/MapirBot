"""Graphiti service for Mapir Memory MCP."""

import logging
import json
import asyncio
from datetime import datetime, timezone
from typing import Any, Optional, cast

from pydantic import BaseModel, Field
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client import openai_base_client as obc, openai_client as oc
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.search.search_config import (
    NodeSearchMethod, 
    NodeSearchConfig, 
    SearchConfig,
    EdgeSearchMethod,
    EdgeSearchConfig,
    EpisodeSearchMethod,
    EpisodeSearchConfig
)
from graphiti_core.search.search_filters import SearchFilters

from .extraction import process_simple_extraction

logger = logging.getLogger(__name__)

# --- Structured Entity Models (Reference Inspired) ---

class Requirement(BaseModel):
    """A Requirement represents a specific need or functionality."""
    node_name: str = Field(..., description='Brief name of the requirement (e.g., "GPU Optimization").')
    project_name: str = Field(..., description='The name of the project.')
    description: str = Field(..., description='Description of the requirement.')

class Preference(BaseModel):
    """A Preference represents a user's expressed like or dislike. 
    IMPORTANT: The node_name should be the OBJECT (Pizzao), NOT the person (Jorge)."""
    node_name: str = Field(..., description='The OBJECT of the preference. Example: "Pizza", "Pádel", "Leer".')
    category: str = Field(..., description="Category (Food, Sport, Hobby, etc.)")
    description: str = Field(..., description='Sentence: "Le gusta [node_name]".')

class Procedure(BaseModel):
    """A Procedure informing the agent how to perform certain actions."""
    node_name: str = Field(..., description='The name of the procedure.')
    description: str = Field(..., description='Brief description of the procedure.')

class UserAction(BaseModel):
    """
    Representa una acción específica realizada por el usuario hacia el agente. 
    Es crucial para entender si el usuario está nutriendo la base de conocimiento 
    o buscando extraer información de ella.

    Instrucciones para identificar y extraer UserActions:
    1. Determinar el flujo: ¿El usuario entrega algo (archivo, dato, instrucción) o pide algo?
    2. Identificar el 'tema': El concepto principal (ej. "Robótica", "Python", "Ventas").
    3. Clasificar el tipo: 'Envío' (si aporta info) o 'Solicitud' (si pregunta o pide tarea).
    4. Referenciar objetos: Si se menciona un archivo, incluir el nombre o tipo.
    5. Extraer la intención: ¿Cuál es el objetivo final de esta acción específica?
    """

    action_type: str = Field(
        ...,
        description="Tipo de interacción: 'Enviar' (el usuario aporta información/documentos) o 'Solicitar' (el usuario hace preguntas o pide analizar datos).",
    )
    topic: str = Field(
        ...,
        description="El tema principal sobre el que versa la acción (ej. 'IA', 'Gastronomía', 'Archivo de ventas').",
    )
    action_summary: str = Field(
        ...,
        description="Resumen breve de lo ocurrido. Ej: 'Envió un PDF sobre visión artificial' o 'Preguntó sobre el funcionamiento del grafo'.",
    )
    intent: str = Field(
        "General",
        description="The objective (Compartir info, Resolver duda, etc.).",
    )
    user_name: str = Field(
        "Usuario",
        description="El nombre del usuario que realiza la acción (ej. Jorge).",
    )

ENTITY_TYPES: dict[str, type[BaseModel]] = {
    'Requirement': Requirement,
    'Preference': Preference,
    'Procedure': Procedure,
    'UserAction': UserAction,
}

# --- Service Implementation ---

class MapirMemoryService:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.client: Optional[Graphiti] = None
        self._lock = asyncio.Lock()
        
        # Queue system for sequential processing per group_id
        self.episode_queues: dict[str, asyncio.Queue] = {}
        self.queue_workers: dict[str, bool] = {}

    async def get_client(self) -> Graphiti:
        async with self._lock:
            if self.client is None:
                logger.info("Initializing Graphiti client...")
                
                # --- APPLY RUNTIME PATCHES ---
                self._apply_patches()
                
                # Setup LLM Client with LangChain Wrapper
                from .langchain_wrapper import LangChainGraphitiClient
                llm_config = LLMConfig(
                    api_key=self.config['llm']['api_key'],
                    base_url=self.config['llm']['base_url'],
                    model=self.config['llm']['model'],
                    temperature=0.0
                )
                llm_client = LangChainGraphitiClient(config=llm_config)
                
                # Setup Embedder Client
                embedder_config = OpenAIEmbedderConfig(
                    api_key=self.config['embedder']['api_key'],
                    base_url=self.config['embedder']['base_url'],
                    embedding_model=self.config['embedder']['model']
                )
                embedder_client = OpenAIEmbedder(embedder_config)

                # Configure Graphiti
                self.client = Graphiti(
                    uri=self.config['graphiti']['neo4j_uri'],
                    user=self.config['graphiti']['neo4j_user'],
                    password=self.config['graphiti']['neo4j_password'],
                    llm_client=llm_client,
                    embedder=embedder_client
                )
                
                logger.info("Graphiti client initialized with LangChain wrapper and patches.")
            return self.client

    # --- Background Processing Logic ---

    async def _process_episode_queue(self, group_id: str):
        """Sequential worker for a specific group_id."""
        logger.info(f"Starting sequential worker for group_id: {group_id}")
        self.queue_workers[group_id] = True
        queue = self.episode_queues[group_id]
        
        try:
            while True:
                task = await queue.get()
                try:
                    await task()
                except Exception as e:
                    logger.error(f"Worker error for group {group_id}: {e}")
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            logger.info(f"Worker for group {group_id} cancelled.")
        finally:
            self.queue_workers[group_id] = False

    async def add_episode_queued(self, name: str, content: str, group_id: str, source: str = "text", source_description: str = ""):
        """Add an episode to the sequential processing queue with a redundancy check."""
        client = await self.get_client()
        
        # --- REDUNDANCY CHECK ---
        # Very basic check: compare with last 3 episodes in this group
        # Real implementation would use embeddings, but this is a good start
        # to satisfy "no sean muy parecidos"
        recent_episodes = await self.search_episodes(content[:100], [group_id], limit=3)
        for ep in recent_episodes:
            # If content is identical or very similar (simple check)
            if content.strip() == ep.content.strip():
                logger.info(f"Skipping redundant episode in group {group_id}: content matches exactly.")
                return 0
        
        # Ensure queue and worker exist
        if group_id not in self.episode_queues:
            self.episode_queues[group_id] = asyncio.Queue()
        
        if not self.queue_workers.get(group_id, False):
            asyncio.create_task(self._process_episode_queue(group_id))

        # Map source string to EpisodeType
        st = EpisodeType.text
        if source.lower() == "message": st = EpisodeType.message
        elif source.lower() == "json": st = EpisodeType.json

        async def task():
            logger.info(f"Processing queued episode: {name} for group {group_id}")
            # Use custom entity types for better extraction
            try:
                await client.add_episode(
                    name=name,
                    episode_body=content,
                    group_id=group_id,
                    source=st,
                    source_description=source_description,
                    reference_time=datetime.now(timezone.utc),
                    entity_types=ENTITY_TYPES
                )
                logger.info(f"Successfully added episode: {name} for group {group_id}")
            except Exception as e:
                logger.error(f"Failed to add episode {name}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        await self.episode_queues[group_id].put(task)
        return self.episode_queues[group_id].qsize()

    # --- Search Methods ---

    async def search_nodes(self, query: str, group_ids: list[str], limit: int = 10):
        client = await self.get_client()
        config = SearchConfig(
            node_config=NodeSearchConfig(
                search_methods=[NodeSearchMethod.cosine_similarity],
                limit=limit
            )
        )
        results = await client.search_(query, config=config, group_ids=group_ids)
        return results.nodes

    async def search_facts(self, query: str, group_ids: list[str], limit: int = 10):
        client = await self.get_client()
        config = SearchConfig(
            edge_config=EdgeSearchConfig(
                search_methods=[EdgeSearchMethod.cosine_similarity],
                limit=limit
            )
        )
        results = await client.search_(query, config=config, group_ids=group_ids)
        return results.edges

    async def search_episodes(self, query: str, group_ids: list[str], limit: int = 10):
        client = await self.get_client()
        config = SearchConfig(
            episode_config=EpisodeSearchConfig(
                search_methods=[EpisodeSearchMethod.bm25],
                limit=limit
            )
        )
        results = await client.search_(query, config=config, group_ids=group_ids)
        return results.episodes

    async def clear_graph(self, group_id: str):
        """Clear graph data. If group_id is 'ALL', wipes everything."""
        client = await self.get_client()
        from graphiti_core.utils.maintenance.graph_data_operations import clear_data
        
        if group_id.upper() == "ALL":
            logger.info("Wiping entire graph database...")
            await clear_data(client.driver, group_ids=None)
        else:
            logger.info(f"Clearing graph data for group: {group_id}")
            await clear_data(client.driver, group_ids=[group_id])
            # Extra safety: wipe anything without a group_id if it's the default group
            if group_id == "main" or group_id == "":
                 await client.driver.execute_query("MATCH (n) WHERE n.group_id IS NULL OR n.group_id = '' DETACH DELETE n")
    async def merge_nodes_by_name(self, group_id: str, primary_name: str, duplicate_name: str):
        """Merge nodes with given names in Neo4j."""
        client = await self.get_client()
        
        # Cypher query to merge nodes
        # This moves all relationships from duplicate to primary and deletes duplicate
        # It also merges properties (primary wins on conflicts)
        query = """
        MATCH (p:Entity {group_id: $group_id, name: $primary_name})
        MATCH (d:Entity {group_id: $group_id, name: $duplicate_name})
        WHERE id(p) <> id(d)
        
        // Move outgoing relationships
        MATCH (d)-[r]->(target)
        CALL apoc.refactor.from(r, p) YIELD input, output
        RETURN count(*)
        """
        # Wait, if APOC isn't available, we need a fallback.
        # Let's use a more universal Cypher for merging relationships.
        
        query_fallback = """
        MATCH (p:Entity {group_id: $group_id, name: $primary_name})
        MATCH (d:Entity {group_id: $group_id, name: $duplicate_name})
        WHERE p.uuid <> d.uuid
        
        // Move outgoing relationships
        MATCH (d)-[r]->(target)
        FOREACH (ignore IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END |
            CREATE (p)-[r2:RELATION]->(target)
            SET r2 = properties(r), r2.fact = r.fact
        )
        DELETE r
        WITH p, d
        
        // Move incoming relationships
        MATCH (source)-[r]->(d)
        FOREACH (ignore IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END |
            CREATE (source)-[r2:RELATION]->(p)
            SET r2 = properties(r), r2.fact = r.fact
        )
        DELETE r
        WITH p, d
        
        // Append summaries if they are distinct
        SET p.summary = p.summary + " " + d.summary
        
        DELETE d
        """
        
        # Graphiti doesn't expose a raw cypher runner easily without accessing the driver
        await client.driver.execute_query(query_fallback, params={
            "group_id": group_id,
            "primary_name": primary_name,
            "duplicate_name": duplicate_name
        })
        
        logger.info(f"Merged nodes '{duplicate_name}' into '{primary_name}' in group {group_id}")

    async def update_node(self, group_id: str, name: str, new_summary: Optional[str] = None, attributes: Optional[dict] = None):
        """Update node summary or attributes in Neo4j."""
        client = await self.get_client()
        
        if new_summary:
            query = "MATCH (n:Entity {group_id: $group_id, name: $name}) SET n.summary = $summary"
            await client.driver.execute_query(query, params={"group_id": group_id, "name": name, "summary": new_summary})
            logger.info(f"Updated summary for node '{name}'")
            
        if attributes:
            # MapirBot could provide a dict of attributes to update
            query = "MATCH (n:Entity {group_id: $group_id, name: $name}) SET n += $props"
            await client.driver.execute_query(query, params={"group_id": group_id, "name": name, "props": attributes})
            logger.info(f"Updated attributes for node '{name}': {attributes}")

    async def invalidate_edge(self, group_id: str, edge_uuid: str):
        """Mark an edge as invalid in Neo4j."""
        client = await self.get_client()
        # Graphiti edges have an 'invalid_at' property. We'll set it to now.
        now = datetime.now(timezone.utc).isoformat()
        query = "MATCH ()-[r {group_id: $group_id, uuid: $uuid}]->() SET r.invalid_at = $now"
        await client.driver.execute_query(query, params={"group_id": group_id, "uuid": edge_uuid, "now": now})
        logger.info(f"Invalidated edge {edge_uuid} in group {group_id}")

    def _apply_patches(self):
        """Patch Graphiti internal logic to improve local LLM performance."""
        import graphiti_core.utils.maintenance.dedup_helpers as dh
        
        # Original entropy check blocks short names from matching exactly
        def patched_resolve_with_similarity(
            extracted_nodes: list[Any],
            indexes: Any,
            state: Any,
        ) -> None:
            """Patched version that allows exact name matching even for short/low-entropy names."""
            for idx, node in enumerate(extracted_nodes):
                normalized_exact = dh._normalize_string_exact(node.name)
                normalized_fuzzy = dh._normalize_name_for_fuzzy(node.name)

                # --- PATCH START: Check exact matches BEFORE entropy check ---
                existing_matches = indexes.normalized_existing.get(normalized_exact, [])
                if len(existing_matches) == 1:
                    match = existing_matches[0]
                    state.resolved_nodes[idx] = match
                    state.uuid_map[node.uuid] = match.uuid
                    if match.uuid != node.uuid:
                        state.duplicate_pairs.append((node, match))
                    continue
                # --- PATCH END ---

                if not dh._has_high_entropy(normalized_fuzzy):
                    state.unresolved_indices.append(idx)
                    continue

                if len(existing_matches) > 1:
                    state.unresolved_indices.append(idx)
                    continue

                shingles = dh._cached_shingles(normalized_fuzzy)
                signature = dh._minhash_signature(shingles)
                candidate_ids: set[str] = set()
                for band_index, band in enumerate(dh._lsh_bands(signature)):
                    candidate_ids.update(indexes.lsh_buckets.get((band_index, band), []))

                best_candidate: Any | None = None
                best_score = 0.0
                for candidate_id in candidate_ids:
                    candidate_shingles = indexes.shingles_by_candidate.get(candidate_id, set())
                    score = dh._jaccard_similarity(shingles, candidate_shingles)
                    if score > best_score:
                        best_score = score
                        best_candidate = indexes.nodes_by_uuid.get(candidate_id)

                if best_candidate is not None and best_score >= dh._FUZZY_JACCARD_THRESHOLD:
                    state.resolved_nodes[idx] = best_candidate
                    state.uuid_map[node.uuid] = best_candidate.uuid
                    if best_candidate.uuid != node.uuid:
                        state.duplicate_pairs.append((node, best_candidate))
                    continue

                state.unresolved_indices.append(idx)
        
        # Apply the patch
        dh._resolve_with_similarity = patched_resolve_with_similarity
        logger.info("Applied patch to Graphiti node deduplication logic (Entropy bypass for exact matches).")
