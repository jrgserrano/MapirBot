import asyncio
import os
import sys

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import create_graph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_mcp_adapters.client import MultiServerMCPClient

async def run_test():
    #mcp_client = MultiServerMCPClient({
    #    "paper_search": {
    #        "command": sys.executable,
    #        "args": ["-m", "paper_search_mcp.server"],
    #        "transport": "stdio"
    #    }
    #})
    
    # Direct usage as recommended by the error message
    #tools = await mcp_client.get_tools()
    
    os.makedirs("database", exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string("database/checkpoints.sqlite") as checkpointer:
        #graph_app = create_graph(tools, checkpointer)
        graph_app = create_graph(checkpointer)
        config = {"configurable": {"thread_id": "cli_test"}}
        
        print("\n--- TEST: ASK QUESTION ---")
        resp = await graph_app.ainvoke({"messages": [("user", "Puedes buscar información sobre el 3D Gaussian Splatting?")]}, config)
        
        #resp = await graph_app.ainvoke({"messages": [("user", "Donde está Irán?")]}, config)

if __name__ == "__main__":
    asyncio.run(run_test())
