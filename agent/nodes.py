from langchain_core.messages import SystemMessage
from agent.llm import llm_main, llm_router, llm_text
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from agent.state import AgentState
from langchain_core.messages import RemoveMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Literal
import re
import json
#from graphiti_core.nodes import EpisodeType
from datetime import datetime, timezone

async def planner(state: AgentState) -> dict:
    
    query = state.get('messages')[-1].content
    state['query'] = query

    for m in state.get('messages'):
        if isinstance(m, HumanMessage):
            print(f"[INFO] User Query: {m.content}")

    system_prompt = f"""
    You are an expert reasoning and planning agent. Your task is to analyze the user's query, decompose the problem, and formulate an abstract plan to retrieve the necessary information from external databases. You must NOT answer the user's question directly.

    You have access to the following tools:
    1. Graphiti: A knowledge graph database. Use this tool to search for specific entities, people, projects, and the structural relationships between them.
    2. ChromaDB: A vector database. Use this tool to search for documents, manuals, descriptions, or unstructured semantic text.

    You must STRICTLY follow this two-step output format:

    Thought: <Write your step-by-step reasoning to solve the user's query. Explain what entities you need to find and what documents you need to search.>
    Abstract Chain: <Generate the exact tool calls using the format [query -ToolName-> variable]. Use variables like y1, y2, y3 to represent the future results.>

    CRITICAL RULES:
    - Never answer the question directly. Only output the "Thought" and the "Abstract Chain".
    - Enclose all tool calls in square brackets.
    - The format inside the brackets MUST be: [what to search -ToolName-> variable].
    - Valid ToolNames are only "Graphiti" and "ChromaDB".
    - If a tool call depends on the result of a previous tool call, you MUST inject the previous variable into the new query (e.g., if you found a person in y1, and need their resume, write [resume of y1 -ChromaDB-> y2]).

    EXAMPLES:

    User: Who is the manager of the Alpha Project and what are their main skills?
    Thought: I need to find the manager of the Alpha Project using the knowledge graph. Once I have the manager's name, I need to search the vector database for documents containing their skills.
    Abstract Chain: First, find the [manager of Alpha Project -Graphiti-> y1]. Then, find the [skills of y1 -ChromaDB-> y2].

    User: What is the operating temperature of the bolometer mentioned in the X-200 manual?
    Thought: I need to search the vector database for the X-200 manual to find the section about the bolometer, and then extract the operating temperature from it. 
    Abstract Chain: Search for [bolometer operating temperature in X-200 manual -ChromaDB-> y1].

    User: Where is the headquarters of Ralph Hefferline's company?
    Thought: I first need to find the company Ralph Hefferline works for in the knowledge graph. Then, I need to find the headquarters location of that specific company in the vector database.
    Abstract Chain: Find the [company of Ralph Hefferline -Graphiti-> y1]. Then, search for the [headquarters of y1 -ChromaDB-> y2].

    User: Hello, how are you?
    Thought: I need to answer the user's question directly.
    Abstract Chain: None


    Now, process the following user query:
    {query}
    """

    messages = [SystemMessage(content=system_prompt)]
    
    response = await llm_router.ainvoke(messages)

    print(f"[INFO] Abstract Plan: {response.content}")

    return {
        "abstract_plan": response.content
    }
    

    

async def final_answer_agent(state: AgentState) -> dict:
    """ This node writes the final answer based on the current state of the agent """

    current_time = state.get("current_time", "Unknown")

    prompt = f"""
    You are MapirBot an AI assistant for Mapir. You have to act as a human assistant following the rules below:
     1. You have to answer in the same language as the user.
     2. You have to answer as short as possible. BE EXTREMELY DRY AND CONCISE. Do not use conversational filler.
     3. You have to answer in a natural way. 
     4. You have to answer in a direct way. Answer the question immediately.
     5. You have to answer in a helpful way. 
    CURRENT TIME: {current_time}
    """

    prompt += "\n\nContext from tools and history:"
    
    # Collect ALL context from tool-related messages since the last HumanMessage
    tool_contexts = []
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, AIMessage) and any(x in m.content for x in ["Knowledge Base Info", "Weather Info", "Web Content Info"]):
            tool_contexts.append(m.content)
        # Also handle ToolMessages if they exist (from MCP tools directly)
        elif hasattr(m, 'tool_call_id'): # Generic check for tool message
             tool_contexts.append(f"Tool Result: {m.content}")

    if tool_contexts:
        for ctx in reversed(tool_contexts):
            prompt += f"\n- {ctx}"
    else:
        prompt += f"\n- {state['messages'][-1].content}"
    
    print(f"[DEBUG] Final Answer Prompt: {prompt}")

    messages = [SystemMessage(content=prompt)]
    
    # Add conversation history (excluding the meta-messages we just synthesized into the prompt)
    for m in state["messages"]:
        if isinstance(m, AIMessage) and any(x in m.content for x in ["Knowledge Base Info", "Weather Info", "Web Content Info"]):
            continue
        messages.append(m)

    response = await llm_text.ainvoke(messages)
    
    # Strip reasoning if present
    response.content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
    
    if response.content:
        print(f"[INFO] Final Answer Generated: {response.content[:50]}...")
    else:
        print("[WARN] Final Answer is EMPTY!")
        response.content = "No he podido procesar una respuesta clara con la información obtenida. ¿Podrías reformular la pregunta?"

    return {
        "messages": [response], 
        "next_step": None, 
        "logs": []
    } 

async def trim_memory(state: AgentState) -> dict:
    """ 
        This node trims the message history to keep only the last N messages, 
        ensuring it always starts with a User message to satisfy LLM requirements.

        It also updates the summary of the conversation.

        Returns:
            dict: A dictionary containing the trimmed messages and the updated summary.
    """
    messages = state["messages"]
    prev_summary = state.get("summary", "")
    window_size = 10
    
    if len(messages) > window_size:
        
        to_remove = len(messages) - window_size

        # Ensure we always start with a HumanMessage after removal
        # We loop until the first message in the remaining list is a HumanMessage
        while to_remove < len(messages) and not isinstance(messages[to_remove], HumanMessage):
            to_remove += 1
            
        if to_remove > 0:

            prompt = f"""
                This is the summary of the previous conversation: {prev_summary}
                You have to update the summary of the conversation whit the messages that are going to be removed.
                Return only the updated summary, without any additional text.
                These are the messages that are going to be removed: {messages[:to_remove]}
            """

            updated_summary = await llm_main.ainvoke(prompt)
            
            return {"messages": [RemoveMessage(id=m.id) for m in messages[:to_remove]], "summary": updated_summary.content}
            
    return {"messages": [], "summary": prev_summary}