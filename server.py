import os
import dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.graph import create_graph

dotenv.load_dotenv()

class UserRequest(BaseModel):
    query: str
    thread_id: str = "default_api_thread"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize MCP client
    app.mcp_client = MultiServerMCPClient({
        "paper_search": {
            "command": "python",
            "args": ["-m", "paper_search_mcp.server"],
            "transport": "stdio"
        }
    })
    
    # Initialize tools
    print("[INFO] Initializing MCP tools...")
    tools = await app.mcp_client.get_tools()
    
    # Ensure database directory exists
    os.makedirs("database", exist_ok=True)
    
    # Initialize checkpointer and graph
    async with AsyncSqliteSaver.from_conn_string("database/checkpoints.sqlite") as checkpointer:
        app.graph_app = create_graph(tools, checkpointer)
        yield
    
    await app.mcp_client.close()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "MapirBot API is running"}

@app.post("/ask")
async def ask_question(request: UserRequest):
    if not hasattr(app, "graph_app"):
        raise HTTPException(status_code=503, detail="AI Agent not initialized")
    
    config = {"configurable": {"thread_id": request.thread_id}}
    final_response = ""

    print("[INFO] User query: ", request.query)
    
    try:
        async for chunk in app.graph_app.astream({"messages": [("user", request.query)]}, config, stream_mode="updates"):
            for node, update in chunk.items():
                print(f"[INFO] Node: {node}")
                if node == "final_answer":
                    if "messages" in update:
                        final_response += update["messages"][-1].content
        
        return {"query": request.query, "response": final_response}
    except Exception as e:
        print(f"[ERROR] Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))