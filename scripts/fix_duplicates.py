import asyncio
import os
import yaml
from pathlib import Path
from services.memory_service import MapirMemoryService

def load_config():
    config_path = Path("mcp_servers/mapir_memory/config/config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

async def fix_duplicates():
    print("=== Mapir Memory Duplicate Fixer ===")
    config = load_config()
    service = MapirMemoryService(config)
    client = await service.get_client()
    
    group_id = "jorge" # Default for chat
    
    # 1. Find duplicate names
    query_find = """
    MATCH (n:Entity {group_id: $group_id})
    WITH n.name as name, count(n) as node_count
    WHERE node_count > 1
    RETURN name, node_count
    """
    
    results = await client.driver.execute_query(query_find, params={"group_id": group_id})
    duplicates = results.records
    
    if not duplicates:
        print("No duplicate entities found.")
        return

    print(f"Found {len(duplicates)} entities with duplicates.")
    
    for record in duplicates:
        name = record["name"]
        count = record["node_count"]
        print(f"Merging {count} nodes for entity: '{name}'...")
        
        # We merge all into the first one. 
        # The service merge_nodes_by_name handles two at a time, we might need multiple passes.
        # But our custom Cypher query actually handles it if we tweak it.
        # For simplicity, we'll just call the service method which is already robust.
        # Wait, the service method only merges two nodes. We can call it n-1 times.
        
        for i in range(count - 1):
             await service.merge_nodes_by_name(group_id, name, name)
             # Wait, my query uses WHERE p.uuid <> d.uuid, so it will find a pair each time.
    
    print("Done! All duplicates merged.")

if __name__ == "__main__":
    os.environ["PYTHONPATH"] = str(Path("mcp_servers/mapir_memory/src").absolute())
    asyncio.run(fix_duplicates())
