import sqlite3
from agent.graph import graph_app
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def list_threads():
    conn = sqlite3.connect("database/checkpoints.sqlite")
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        threads = [row[0] for row in cursor.fetchall()]
        return threads
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

def view_memory(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state = graph_app.get_state(config)
    
    if not state or not state.values:
        print(f"\n[WARN] No state found for thread: {thread_id}")
        return

    values = state.values
    messages = values.get("messages", [])
    summary = values.get("summary", "No summary available.")
    next_step = values.get("next_step", "None")

    print(f"\n" + "="*50)
    print(f" MEMORY FOR THREAD: {thread_id}")
    print("="*50)
    print(f"\n[SUMMARY]\n{summary}")
    print(f"\n[NEXT STEP]: {next_step}")
    print(f"\n[MESSAGES]:")
    
    for i, msg in enumerate(messages):
        role = "???"
        if isinstance(msg, HumanMessage): role = "HUMAN"
        elif isinstance(msg, AIMessage): role = "BOT"
        elif isinstance(msg, SystemMessage): role = "SYSTEM"
        
        content = msg.content
        if len(content) > 200:
            content = content[:200] + "..."
            
        print(f" {i+1}. [{role}]: {content}")
    print("="*50 + "\n")

if __name__ == "__main__":
    threads = list_threads()
    if not threads:
        print("No memory records found in the database.")
    else:
        print(f"Available threads: {threads}")
        for tid in threads:
            view_memory(tid)
