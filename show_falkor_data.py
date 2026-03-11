import asyncio
from falkordb.asyncio import FalkorDB
import os

async def main():
    print("Connecting to FalkorDB at redis://localhost:6379...")
    db = FalkorDB(host='localhost', port=6379)
    graph = db.select_graph('default_db')
    
    try:
        print("\n" + "="*50)
        # Count nodes
        res = await graph.query("MATCH (n) RETURN count(n)")
        print(f"Total nodes in database: {res.result_set[0][0]}")
        
        print("\n--- GRAPHITI EPISODES (RAW MEMORIES) ---")
        res = await graph.query("MATCH (n:Episodic) RETURN n.name, n.content LIMIT 10")
        for row in res.result_set:
            print(f"- Episode: {row[0]}")
            print(f"  Content: {row[1][:100]}...")
            
        print("\n--- GRAPHITI EXTRACTED ENTITIES (NODES) ---")
        res = await graph.query("MATCH (n:Entity) RETURN labels(n), n.name, n.summary LIMIT 20")
        for row in res.result_set:
            labels = row[0]
            name = row[1]
            summary = row[2]
            print(f"- {name} (Type: {labels})")
            if summary:
                print(f"  Summary: {summary}")
                
        print("\n--- GRAPHITI FACTS (EDGES/RELATIONSHIPS) ---")
        res = await graph.query("MATCH (n)-[r:ENTITY_EDGE]->(m) RETURN n.name, r.relation, m.name LIMIT 20")
        for row in res.result_set:
            print(f"- Fact: {row[0]} -> {row[1]} -> {row[2]}")
            
    except Exception as e:
        print(f"Error querying FalkorDB: {e}")
    finally:
        await db.aclose()

if __name__ == "__main__":
    asyncio.run(main())
