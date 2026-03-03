import requests
import os
import dotenv
import aiohttp
from slack_bolt.app.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from tools.knowledge_base import knowledge_base_update
import io
from pypdf import PdfReader
import asyncio
from datetime import datetime, timezone

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

    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "query": clean_query,
                "thread_id": channel_id
            }
            async with session.post("http://127.0.0.1:8000/ask", json=payload) as api_response:
                if api_response.status == 200:
                    data = await api_response.json()
                    response = data.get("response", "Lo siento, hubo un problema procesando tu petición.")
                else:
                    response = f"_Error de API: HTTP {api_response.status}_"
                    print(f"[ERROR] API returned {api_response.status}: {await api_response.text()}")
        
        await say(text=response)
    except Exception as e:
        print(f"[ERROR] Failed to contact API: {e}")
        await say(text="_Error contactando al servidor de MapirBot._")

async def main():
    handler = AsyncSocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    print("[INFO] Starting MapirBot Slack App (API Client mode)...")
    await handler.start_async()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[INFO] Bot stopped by user.")