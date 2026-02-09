import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import graph_app

# State with thread_id for short-term memory
config = {"configurable": {"thread_id": "knowledge_test"}}

# 1. TURN 1: INTRODUCE
print("--- TURN 1: INTRODUCE ---")
resp = graph_app.invoke({"messages": [("user", "Hi! My name is Jorge.")]}, config)
print("User: Hi! My name is Jorge.")
print(f"Agent: {resp.get('messages')[-1].content}")

# 2. TURN 2: VERIFY MEMORY
print("\n--- TURN 2: VERIFY MEMORY ---")
resp = graph_app.invoke({"messages": [("user", "What is my name?")]}, config)
print("User: What is my name?")
print(f"Agent: {resp.get('messages')[-1].content}")

print("\n--- TEST COMPLETE ---")

# Save graph visualization to a file
try:
    with open("graph.png", "wb") as f:
        f.write(graph_app.get_graph().draw_mermaid_png())
    print("Graph visualization saved to graph.png")
except Exception as e:
    print(f"Could not save graph image: {e}")