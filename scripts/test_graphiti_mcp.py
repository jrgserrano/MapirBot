import asyncio
import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import logging

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

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("MapirChat")

def clean_messages(msgs, limit=12):
    """Clean and truncate messages to stay within local LLM context limits."""
    result = []
    
    # Always keep system message
    if msgs and isinstance(msgs[0], SystemMessage):
        result.append(msgs[0])
    
    # Get last N messages
    non_system = [m for m in msgs if not isinstance(m, SystemMessage)]
    
    # We want to find a safe cutoff point that starts with a HumanMessage
    start_idx = max(0, len(non_system) - limit)
    
    # Backtrack or forward-track to the nearest HumanMessage
    while start_idx < len(non_system) and not isinstance(non_system[start_idx], HumanMessage):
        start_idx += 1
        
    last_n = non_system[start_idx:]
    
    # Final cleaning of assistant messages for display/context efficiency
    for i, m in enumerate(last_n):
        if isinstance(m, AIMessage) and m.content:
            # Strip thinking
            content = re.sub(r'<think>.*?</think>', '', m.content, flags=re.DOTALL).strip()
            if not content and m.tool_calls:
                content = "Calling tools..."
            last_n[i] = AIMessage(content=content, tool_calls=m.tool_calls, id=m.id)
            
    result.extend(last_n)
    return result

async def run_chat():
    print(f"{Colors.HEADER}{Colors.BOLD}=== Mapir Memory Interactive Chat ==={Colors.ENDC}")
    print(f"{Colors.CYAN}Commands: /status, /clear, /visualize, /exit{Colors.ENDC}\n")

    # Setup Environment
    os.environ["OPENAI_API_KEY"] = "lm-studio"
    os.environ["OPENAI_BASE_URL"] = "http://localhost:1234/v1"
    
    python_executable = "/home/ubuntu/MapirBot/venv/bin/python"
    server_path = str(Path("mcp_servers/mapir_memory/src/mapir_memory_server.py").absolute())
    
    mcp_env = os.environ.copy()
    mcp_env["PYTHONPATH"] = str(Path("mcp_servers/mapir_memory/src").absolute())

    mcp_client = MultiServerMCPClient(
        {
            "mapir_memory": {
                "transport": "stdio",
                "command": python_executable,
                "args": [server_path],
                "env": mcp_env
            }
        }
    )

    async with mcp_client.session("mapir_memory") as session:
        from langchain_mcp_adapters.tools import load_mcp_tools
        tools = await load_mcp_tools(session)
        
        llm = ChatOpenAI(
            model="mistral-nemo-12b-instruct-2407",
            temperature=0
        )
        
        system_prompt = (
            "Eres MapirBot, un asistente avanzado con memoria a largo plazo. "
            "Tu objetivo es ayudar al usuario y mantener un grafo de conocimiento coherente y actualizado."
            "\n\nPROCESO DE PENSAMIENTO OBLIGATORIO:"
            "\n1. Analiza el mensaje: ¿Es un hecho? ¿Una intención? ¿Una búsqueda?"
            "\n2. Si el usuario te pregunta por algo que deberías saber (ej. '¿Qué me gusta?'), usa 'search_nodes' o 'search_facts' MÁXIMO 2 VECES con diferentes términos."
            "\n3. REGLA DE ORO: Si tras 2 búsquedas no encuentras información clara, DEBES responder honestamente que no tienes esa información guardada aún. No entres en bucles infinitos de búsqueda."
            "\n4. Si el usuario te da información nueva, usa 'add_memory' antes de responder."
            "\n\nESTRATEGIA DE MEMORIA:"
            "\n- HECHOS: Guarda Preferencias, Requisitos o Procedimientos."
            "\n- INTENCIONES: Registra planes con 'UserAction'."
            "\n- CONFLICTOS: Invalidar antiguos hechos es prioritario si hay cambios."
            "\n\nCRÍTICO: Usa siempre group_id='jorge' para todas las herramientas de memoria."
            "\nResponde siempre en el mismo idioma que el usuario."
        )
        
        agent = create_react_agent(llm, tools, prompt=system_prompt)
        messages = [] # System prompt is handled by prompt

        user_id = "jorge"

        while True:
            try:
                user_input = input(f"{Colors.GREEN}{Colors.BOLD}Tú > {Colors.ENDC}").strip()
                if not user_input: continue
                
                if user_input.lower() == '/exit':
                    break
                elif user_input.lower() == '/clear':
                    await session.call_tool("clear_graph", {"group_id": "ALL"})
                    messages = []
                    print(f"{Colors.YELLOW}Base de datos completa y sesión limpiadas.{Colors.ENDC}")
                    continue
                elif user_input.lower() == '/fix':
                    print(f"{Colors.YELLOW}Buscando y fusionando duplicados...{Colors.ENDC}")
                    os.system(f"PYTHONPATH=mcp_servers/mapir_memory/src {python_executable} scripts/fix_duplicates.py")
                    continue
                elif user_input.lower() == '/status':
                    res = await session.call_tool("get_status", {"group_id": user_id})
                    print(f"{Colors.BLUE}Status: {res.content[0].text}{Colors.ENDC}")
                    continue
                elif user_input.lower() == '/visualize':
                    print(f"{Colors.YELLOW}Generando visualización...{Colors.ENDC}")
                    os.system(f"{python_executable} scripts/visualize_graph.py")
                    continue

                # Add user message to history
                messages.append(HumanMessage(content=user_input))
                
                # Truncate/Clean for LLM
                inputs = {"messages": clean_messages(messages)}
                
                print(f"{Colors.CYAN}MapirBot pensando...{Colors.ENDC}", end="\r")
                
                full_response = ""
                # Added recursion_limit to prevent infinite loops at the execution level
                async for event in agent.astream(inputs, config={"recursion_limit": 15}):
                    for node, value in event.items():
                        if "messages" in value:
                            last_msg = value["messages"][-1]
                            if node == "agent":
                                # Extract content from response
                                content = last_msg.content
                                if content:
                                    # Strip thinking from display
                                    display_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                                    if display_content:
                                        if not full_response:
                                            print(" " * 30, end="\r") # Clear thinking line
                                            print(f"{Colors.BLUE}{Colors.BOLD}MapirBot > {Colors.ENDC}", end="")
                                        print(display_content, flush=True)
                                        full_response += display_content
                                
                                # Add to real message history
                                messages.append(last_msg)
                            elif node == "tools":
                                for m in value["messages"]:
                                    print(f"{Colors.YELLOW}[Tool: {m.name}] {m.content[:100]}...{Colors.ENDC}")
                                    messages.append(m)
                
                if not full_response:
                    print(f"{Colors.RED}No hubo respuesta.{Colors.ENDC}")
                
                print()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.ENDC}")

    print(f"\n{Colors.HEADER}¡Hasta luego!{Colors.ENDC}")

if __name__ == "__main__":
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        pass
