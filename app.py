import requests
import os
import dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agent.graph import graph_app
from tools.knowledge_base import knowledge_base_update
import io
from pypdf import PdfReader

dotenv.load_dotenv()

app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    process_before_response=True
)

proccessed_events = set()

# @app.event("app_mention")
# def handle_app_mention_events(event, say, client):
#     handle_message_events(event, say, client)

@app.event("message")
def handle_message_events(event, say, client):
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
    # thread_id = event.get("thread_ts") or event.get("ts")

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
                    status = knowledge_base_update.invoke({
                        "text" : extracted_text,
                        "source_name" : file_name
                    })
                    print(f"[INFO] Knowledge base updated: {file_name} - {status}")
                else:
                    print(f"[WARN] No text extracted from {file_name} (Type: {content_type})")

    is_dm = event.get("channel_type") == "im"
    bot_user_id = client.auth_test()["user_id"]
    was_mentioned = f"<@{bot_user_id}>" in user_query

    if not is_dm and not was_mentioned:
        return

    clean_query = user_query.replace(f"<@{bot_user_id}>", "").strip()
    
    config = {"configurable": {"thread_id": channel_id}}

    response = ""

    print(f"[INFO] User query: {clean_query}")

    # Use a thread_id if we want to reply in thread
    thread_ts = event.get("ts")

    for chunk in graph_app.stream({"messages": [("user", clean_query)]}, config, stream_mode="updates"):
        for node, update in chunk.items():
            print(f"[INFO] Node: {node}")
            
            if node == "knowledge_base":
                say(text="_Checking the database..._", thread_ts=thread_ts)
            
            if node == "final_answer":
                if "messages" in update:
                    response += update["messages"][-1].content
        
    if response:
        say(text=response, thread_ts=thread_ts)
        

if __name__ == "__main__":
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    print("[INFO] Starting MapirBot...")
    handler.start()