import requests
import os
import dotenv
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.graph import create_graph
from tools.knowledge_base import knowledge_base_update
import io
from pypdf import PdfReader
import asyncio
from datetime import datetime, timezone
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

dotenv.load_dotenv()

app = AsyncApp(
    token=os.getenv("SLACK_BOT_TOKEN"),
    process_before_response=True
)

proccessed_events = set()
graph_app = None

@app.event("message")
async def handle_message_events(event, say, client):
    event_id = event.get("client_msg_id")

    if not event_id or event_id in proccessed_events:
        return
    if event.get("subtype") == "bot_message":
        return

    proccessed_events.add(event_id)
    if len(proccessed_events) > 100:
        proccessed_events.clear()

    user_query = event.get("text") or ""
    channel_id = event.get("channel")

    if "files" in event:
        for file in event["files"]:
            file_url = file.get("url_private_download")
            file_name = file.get("name")
            
            if not file_url:
                continue

            response = requests.get(
                file_url,
                headers={"Authorization": f"Bearer {os.getenv('SLACK_BOT_TOKEN')}"},
                stream=True
            )

            if response.status_code == 200:
                extracted_text = ""
                content_type = response.headers.get("Content-Type", "")
                
                # Handle PDF
                if "application/pdf" in content_type or file_name.lower().endswith(".pdf"):
                    try:
                        reader = PdfReader(io.BytesIO(response.content))
                        extracted_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    except Exception as e:
                        print(f"[ERROR] PDF extraction failed for {file_name}: {e}")
                
                # Handle Text-based files
                elif any(file_name.lower().endswith(ext) for ext in [".txt", ".py", ".cpp", ".h", ".md", ".json"]):
                    extracted_text = response.text
                
                if extracted_text:
                    status = await knowledge_base_update.ainvoke({
                        "text" : extracted_text,
                        "source_name" : file_name
                    })
                    print(f"[INFO] Knowledge base updated: {file_name} - {status}")
                else:
                    print(f"[WARN] No text extracted from {file_name} (Type: {content_type})")

    is_dm = event.get("channel_type") == "im"
    auth_info = await client.auth_test()
    bot_user_id = auth_info["user_id"]
    was_mentioned = f"<@{bot_user_id}>" in user_query

    if not is_dm and not was_mentioned:
        return

    clean_query = user_query.replace(f"<@{bot_user_id}>", "").strip()
    config = {"configurable": {"thread_id": channel_id}}
    
    # Inject current time for temporal reasoning
    current_time_str = datetime.now(timezone.utc).isoformat()
    
    response = ""

    print(f"[INFO] User query: {clean_query} (Time: {current_time_str})")

    if not graph_app:
        await say(text="_Error: AI Agent not initialized._")
        return

    async for chunk in graph_app.astream(
        {"messages": [("user", clean_query)], "current_time": current_time_str}, 
        config, 
        stream_mode="updates"
    ):
        for node, update in chunk.items():
            print(f"[INFO] Node: {node}")
            
            if node == "supervisor":
                print(f"[INFO] Supervisor: {update['next_step']}")

            if node == "knowledge_base":
                await say(text="_Checking the database..._")
            
            if node == "weather":
                await say(text="_Checking the weather..._")
            
            if node == "final_answer":
                if "messages" in update:
                    answer_text = update["messages"][-1].content
                    print(f"[INFO] Final Answer Generated: {answer_text[:50]}...")
                    response += answer_text
        
    if response:
        await say(text=response)
        

async def main():
    global graph_app
    
    mcp_client = MultiServerMCPClient({
        "paper_search" : {
            "command" : "python",
            "args" : ["-m", "paper_search_mcp.server"],
            "transport" : "stdio"
        }
    })

    print("[INFO] Initializing MCP tools...")
    tools = await mcp_client.get_tools()
    print(f"[INFO] Retrieved {len(tools)} tools from MCP.")
    
    # Ensure database directory exists
    os.makedirs("database", exist_ok=True)
    
    async with AsyncSqliteSaver.from_conn_string("database/checkpoints.sqlite") as checkpointer:
        graph_app = create_graph(tools, checkpointer)
        print("[INFO] LangGraph initialized with AsyncSqliteSaver.")

        handler = AsyncSocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
        print("[INFO] Starting MapirBot...")
        await handler.start_async()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[INFO] Bot stopped by user.")