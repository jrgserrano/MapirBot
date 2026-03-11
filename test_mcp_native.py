import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    print("Testing Graphiti MCP Server natively...")
    
    server_config = {
        "graphiti": {
            "command": "/Users/jorseme/Desktop/MapirBot/V2/venv/bin/python",
            "args": [
                "/Users/jorseme/Desktop/MapirBot/V2/mcp_servers/graphiti/mcp_server/src/graphiti_mcp_server.py",
                "--transport", "stdio"
            ],
            "transport": "stdio"
        }
    }
    
    # Pass environment variables to the MCP server
    env = os.environ.copy()
    env.update({
        "OPENAI_API_KEY": "lm-studio",
        "OPENAI_BASE_URL": "http://127.0.0.1:1234/v1",
        "MODEL_NAME": "google/gemma-3-4b",
        "EMBEDDER_MODEL": "text-embedding-nomic-embed-text-v1.5",
        "DATABASE_PROVIDER": "neo4j",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "neo4j_changed",
        "CONFIG_PATH": "/Users/jorseme/Desktop/MapirBot/V2/mcp_servers/graphiti/mcp_server/config/config.yaml"
    })
    
    try:
        # Initialize client (langchain-mcp-adapters >= 0.1.0 approach)
        print("Starting client...")
        client = MultiServerMCPClient(server_config=server_config, env=env)
        
        # We need to manually initialize/connect if there are setup methods
        # However, for MultiServerMCPClient we might just be able to use tools directly after instantiating it
        tools = client.get_tools()
        print(f"Got {len(tools)} tools")
        
        print("\n1. Executing Add Memory Tool...")
        for tool in tools:
            if tool.name == "add_memory":
                result = await tool.ainvoke({
                    "name": "Prueba Nativa",
                    "episode_body": "María vive en Madrid y tiene 3 gatos.",
                    "source": "text"
                })
                print(f"Add memory result: {result}")
                break
                
        print("\n2. Executing Flush Queue Tool...")
        for tool in tools:
            if tool.name == "flush_queue":
                # We won't pass group_ids to let it flush the default group 'main'
                result = await tool.ainvoke({})
                print(f"Flush queue result: {result}")
                break
                
        print("\n3. Finished Native test!")
        
    except Exception as e:
        print(f"Exception during test: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    asyncio.run(main())
