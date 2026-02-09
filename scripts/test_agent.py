import sys
import os
# Añadimos la raíz del proyecto al path para poder importar los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import graph_app
from langchain_core.messages import HumanMessage

def run_test_console():
    print("--- 🤖 Mapir Agent Console Test ---")
    print("Escribe 'salir' para terminar.\n")
    
    # ID de hilo para que la memoria de corto plazo funcione en local
    config = {"configurable": {"thread_id": "test_session_1"}}

    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            break

        # Enviamos el mensaje al grafo de LangGraph
        for event in graph_app.stream(
            {"messages": [HumanMessage(content=user_input)]}, 
            config
        ):
            # Aquí verás el "pensamiento" de los agentes
            for node, values in event.items():
                print(f"\n[Nodo: {node}]")
                last_msg = values["messages"][-1]
                print(f"Mapir: {last_msg.content}")

if __name__ == "__main__":
    run_test_console()