import asyncio
import os
import sys
import re
from pathlib import Path
from datetime import datetime

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_mcp_adapters.tools import load_mcp_tools

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import create_graph
from agent.state import AgentState

# Configure terminal colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

async def main():
    print(f"{Colors.HEADER}{Colors.BOLD}=== MapirBot Agent Interactive Chat ==={Colors.ENDC}")
    print(f"Commands: /exit, /visualize\n")

    # Setup MCP
    python_executable = sys.executable
    server_path = os.path.abspath("mcp_servers/mapir_memory/src/mapir_memory_server.py")
    
    mcp_client = MultiServerMCPClient({
        "mapir_memory": {
            "command": python_executable,
            "args": [server_path],
            "transport": "stdio"
        }
    })
    
    os.makedirs("database", exist_ok=True)
    db_path = "database/checkpoints.sqlite"
    
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        async with mcp_client.session("mapir_memory") as session:
            print(f"{Colors.YELLOW}Cargando herramientas del MCP...{Colors.ENDC}")
            tools = await load_mcp_tools(session)
            print(f"{Colors.GREEN}{len(tools)} herramientas cargadas.{Colors.ENDC}")
            
            graph_app = create_graph(checkpointer, tools=tools)
            config = {"configurable": {"thread_id": "interactive_test"}}
            
            while True:
                try:
                    user_input = input(f"{Colors.GREEN}{Colors.BOLD}Tú > {Colors.ENDC}").strip()
                    if not user_input: continue
                    
                    if user_input.lower() == '/exit':
                        break
                    elif user_input.lower() == '/visualize':
                        print(f"{Colors.YELLOW}Generando visualización...{Colors.ENDC}")
                        os.system(f"{python_executable} scripts/visualize_graph.py")
                        continue
                    elif user_input.lower() == '/clear':
                        clear_tool = next((t for t in tools if t.name == "clear_graph"), None)
                        if clear_tool:
                            # Try to get user name from the last 'Name: Query' format if possible
                            # For simplicity, we can also just ask or use 'usuario'
                            print(f"{Colors.YELLOW}Limpiando grafo para 'usuario'...{Colors.ENDC}")
                            res = await clear_tool.ainvoke({"group_id": "ALL"})
                            print(f"{Colors.GREEN}{res}{Colors.ENDC}")
                        else:
                            print(f"{Colors.RED}Herramienta 'clear_graph' no encontrada.{Colors.ENDC}")
                        continue

                    print(f"{Colors.CYAN}MapirBot pensando...{Colors.ENDC}", end="\r")
                    
                    async for event in graph_app.astream(
                        {"messages": [HumanMessage(content=user_input)]}, 
                        config, 
                        stream_mode="updates"
                    ):
                        for node, update in event.items():
                            if update and "messages" in update:
                                msgs = update["messages"]
                                if isinstance(msgs, list) and len(msgs) > 0:
                                    last_msg = msgs[-1]
                                    if isinstance(last_msg, AIMessage):
                                        # Clear the "thinking" message with ANSI escape
                                        sys.stdout.write("\033[K") 
                                        print(f"{Colors.GREEN}{Colors.BOLD}MapirBot > {Colors.ENDC}{last_msg.content}\n")
                            
                            if node == "planner":
                                print(f"{Colors.BLUE}[Planner]{Colors.ENDC} Analizando...")
                            elif node == "executor":
                                print(f"{Colors.BLUE}[Executor]{Colors.ENDC} Ejecutando...")
                            elif node == "log_interaction":
                                if "steps" in update:
                                    for step in update["steps"]:
                                        if "Background Logging" in step:
                                            # Use a distinct color for background logging info
                                            print(f"{Colors.BLUE}[Log]{Colors.ENDC} {step}")
                                        else:
                                            print(f"{Colors.YELLOW}[Log]{Colors.ENDC} {step}")

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"{Colors.RED}Error: {e}{Colors.ENDC}")

    print(f"\n{Colors.YELLOW}Sesión finalizada.{Colors.ENDC}")

if __name__ == "__main__":
    asyncio.run(main())
