import asyncio
import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Set up env variables for LM studio & Graphiti
    os.environ["OPENAI_API_KEY"] = "ollama"
    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["MODEL_NAME"] = "qwen3.5:4b"  # 4B params: Reasonably capable main model, used for complex tasks
    os.environ["SMALL_MODEL_NAME"] = "llama3.2:1b"  # 1B params: Extremely fast, used by Graphiti for simple attribute extraction
    os.environ["EMBEDDER_MODEL"] = "mxbai-embed-large"
    os.environ["EMBEDDER_DIMENSIONS"] = "1024"
    
    # Graphiti specific setup
    os.environ["DATABASE_PROVIDER"] = "falkordb"
    os.environ["FALKORDB_URI"] = "redis://localhost:6379"
    os.environ["CONFIG_PATH"] = str(Path("mcp_servers/graphiti/mcp_server/config/config.yaml").absolute())

    logger.info("Initializing MCP Client with Graphiti...")
    
    # mcp_server/src/graphiti_mcp_server.py is what graphiti exposes
    server_script_path = str(Path("mcp_servers/graphiti/mcp_server/src/graphiti_mcp_server.py").absolute())
    python_executable = sys.executable

    # Setup the MCP client connecting to the Graphiti MCP server
    mcp_client = MultiServerMCPClient(
        {
            "graphiti": {
                "transport": "stdio",
                "command": python_executable,
                "args": [server_script_path, "--transport", "stdio"],
                "env": os.environ.copy() # important to pass the API keys and everything
            }
        }
    )
    
    try:
        from langchain_mcp_adapters.tools import load_mcp_tools
        
        # Get tools from the server using a persistent context session
        logger.info("Starting a persistent session with Graphiti MCP server...")
        async with mcp_client.session("graphiti") as session:
            tools = await load_mcp_tools(session)
            logger.info(f"Got {len(tools)} tools: {[t.name for t in tools]}")
            
            # Initialize an LLM
            llm = ChatOpenAI(
                model="qwen3.5:4b",
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                temperature=0
            )

            # Clear existing graph data for a fresh test run
            logger.info("Clearing existing graph to avoid duplicate entities from previous test runs...")
            clear_tool = next((t for t in tools if t.name == "clear_graph"), None)
            if clear_tool:
                await clear_tool.ainvoke({})
                logger.info("Graph cleared.")
            
            # Setup LangGraph ReAct agent
            logger.info("Setting up LangGraph Agent...")
            system_prompt = (
                "You are an AI assistant. You have access to a long-term memory graph (Graphiti). "
                "CRITICAL: You MUST proactively use the 'add_memory' tool to record any personal facts, "
                "preferences, or important events mentioned by the user during the chat, "
                "even if they do not explicitly ask you to save it. "
                "CRITICAL: If a user shares a file or link with you, you MUST proactively use the 'add_memory' tool "
                "to record that the specific user shared the specific file with you, including what the file is about if possible."
            )
            # Use create_agent instead of create_react_agent to avoid deprecation warning
            from langchain.agents import create_agent
            agent = create_react_agent(llm, tools)
            
            # 1. First interaction: Save information
            user_input_1 = "Hola, soy Jorge y últimamente me encanta comer marisco."
            logger.info("\n" + "="*50)
            logger.info(f"Interaction 1 (Save): {user_input_1}")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input_1}
            ]
            
            async for chunk in agent.astream({"messages": messages}):
                if "agent" in chunk:
                    logger.info(f"Agent Action: {chunk['agent']['messages'][-1].content}")
                if "tools" in chunk:
                    logger.info(f"Tool Execution Results: {chunk['tools']['messages'][-1].content}")
                    
            # We removed the intermediate flush_queue tool call here to allow Graphiti to process 
            # memory asynchronously in the background. The LLM extraction is handled by Graphiti without
            # blocking the main conversational flow.
            
            # 2. Third interaction: Retrieve information (Wait briefly before retrieving)
            logger.info("\n" + "="*50)
            logger.info("Waiting 15 seconds to give background memory processing time before testing retrieval...")
            await asyncio.sleep(15)
            
            user_input_3 = "¿Recuerdas qué comida me gustaba?"
            logger.info("\n" + "="*50)
            logger.info(f"Interaction 2 (Retrieve): {user_input_3}")
            
            # Append to conversation history
            messages.append({"role": "user", "content": user_input_3})
            
            async for chunk in agent.astream({"messages": messages}):
                if "agent" in chunk:
                    logger.info(f"Agent Action: {chunk['agent']['messages'][-1].content}")
                if "tools" in chunk:
                    logger.info(f"Tool Execution Results: {chunk['tools']['messages'][-1].content}")

            # 3. Fourth interaction: New info mapping to existing entity (No explicit save command)
            user_input_4 = "Jorge te ha compartido por el chat un archivo sobre gaussian splatting."
            logger.info("\n" + "="*50)
            logger.info(f"Interaction 3 (Save File Status): {user_input_4}")
            
            messages.append({"role": "user", "content": user_input_4})
            
            async for chunk in agent.astream({"messages": messages}):
                if "agent" in chunk:
                    logger.info(f"Agent Action: {chunk['agent']['messages'][-1].content}")
                if "tools" in chunk:
                    logger.info(f"Tool Execution Results: {chunk['tools']['messages'][-1].content}")
                    
            # Finally, at the very end of the script before it exits, we must flush the queue
            # so that we do not lose the second background memory processing task.
            logger.info("\n" + "="*50)
            logger.info("Script Ending: Forcing backend flush via direct tool call to ensure all memories are saved...")
            flush_tool = next((t for t in tools if t.name == "flush_queue"), None)
            if flush_tool:
                result = await flush_tool.ainvoke({})
                logger.info(f"Final Flush Queue Result: {result}")
            else:
                logger.error("flush_queue tool not found!")

            logger.info("\n" + "="*50)
            logger.info("Finished interactions!")
    except Exception as e:
        import traceback
        logger.error(f"Error during execution: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())
