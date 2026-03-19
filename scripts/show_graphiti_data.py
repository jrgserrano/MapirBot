import asyncio
from neo4j import AsyncGraphDatabase

async def main():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "neo4j_changed"
    
    print(f"Connecting to Neo4j at {uri}...")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    
    try:
        async with driver.session() as session:
            # Check if there is ANY data at all
            result = await session.run("MATCH (n) RETURN count(n) as count")
            record = await result.single()
            total_nodes = record["count"]
            print(f"\nTotal nodes in database: {total_nodes}")
            
            if total_nodes == 0:
                print("The database is completely empty. Note: The previous LangGraph test script ")
                print("may have terminated the MCP server before the background extraction queue finished processing.")
                return

            print("\n" + "="*50)
            print("--- GRAPHITI EPISODES (RAW MEMORIES) ---")
            result = await session.run("""
                MATCH (e:Episodic) 
                RETURN e.name as name, e.content as content
                LIMIT 10
            """)
            async for record in result:
                print(f"- Episode: {record['name']}")
                print(f"  Content: {record['content'][:150]}...")
                
            print("\n" + "="*50)
            print("--- GRAPHITI EXTRACTED ENTITIES (NODES) ---")
            result = await session.run("""
                MATCH (n:Entity) 
                RETURN labels(n) as labels, n.name as name, n.summary as summary 
                LIMIT 20
            """)
            async for record in result:
                print(f"- {record['name']} (Type: {record['labels']})")
                if record['summary']:
                    print(f"  Summary: {record['summary']}")
                
            print("\n" + "="*50)
            print("--- GRAPHITI FACTS (EDGES/RELATIONSHIPS) ---")
            result = await session.run("""
                MATCH (n:Entity)-[r:RELATES_TO]->(m:Entity) 
                RETURN n.name as source, m.name as target, r.fact as fact 
                LIMIT 20
            """)
            async for record in result:
                print(f"- Fact: {record['source']} -> {record['fact']} -> {record['target']}")
                
    finally:
        await driver.close()

if __name__ == "__main__":
    asyncio.run(main())
