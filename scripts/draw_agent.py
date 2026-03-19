import asyncio
import sys
sys.path.append(".")
from agent.graph import create_graph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def main():
    # Usamos 'async with' para manejadores de contexto asíncronos
    async with AsyncSqliteSaver.from_conn_string("database/checkpoints.sqlite") as checkpointer:
        # Suponiendo que create_graph acepta el checkpointer
        graph_app = create_graph(checkpointer)
        
        # Generar la imagen
        with open("graph.png", "wb") as f:
            f.write(graph_app.get_graph().draw_mermaid_png())
    
    print("Grafo generado exitosamente en graph.png")

if __name__ == "__main__":
    asyncio.run(main())