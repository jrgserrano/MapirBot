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
import asyncio
from datetime import datetime, timezone

async def worker(state: AgentState) -> dict:

    user_input = state.get('messages')[-1].content
    # Fallback to existing user_name in state if not detected in this message
    user_name = state.get('user_name', "Usuario")
    
    # Extract user name if format is "Name: User Input"
    name_match = re.match(r'^([^:]+):\s*(.*)$', user_input)
    if name_match:
        user_name = name_match.group(1).strip()
        user_input = name_match.group(2).strip()
        print(f"[INFO] Detected User: {user_name}")

    # Ensure state is updated correctly
    state['user_input'] = user_input
    state['user_name'] = user_name

    chat_summary = state.get('chat_summary', "")

    system_prompt = f"""
    You are a Context Worker Agent. Your task is to read the current user input and the summary of the previous conversation, and generate a new, updated summary that includes both. 

    Here is the summary of the previous conversation: 
    {chat_summary}

    Current user input: 
    {user_input}

    INSTRUCTIONS:
    Generate a concise summary of the most important facts, user details, and the core intent of the current user input. Later, this summary will be used by the Manager Agent to answer the user or store information. Do not answer the user directly, only output the updated summary.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    
    response = await llm_router.ainvoke(messages)
    content = response.content

    # Extract reasoning from Thought: or <think> tags
    reasoning = ""
    reasoning_match = re.search(r'(?:Thought:|<think>)\s*(.*?)(?=\nAbstract Chain:</think>|$)', content, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Extract plan from Abstract Chain:
    summary = content
    plan_match = re.search(r'Abstract Chain:\s*(.*)', content, re.DOTALL | re.IGNORECASE)
    if plan_match:
        summary = plan_match.group(1).strip()

    print(f"[DEBUG] Planner Reasoning: {reasoning[:100]}...")
    print(f"[DEBUG] Chat Summary: {summary}")

    return {
        "reasoning": reasoning,
        "user_name": user_name,
        "chat_summary": summary,
        "user_input": user_input
    }
    
async def planner(state: AgentState) -> dict:
    
    chat_summary = state.get('chat_summary', "")
    user_name = state.get('user_name', "Usuario")
    user_input = state.get('user_input', "")

    print(f"[DEBUG] Chat Summary: {chat_summary}")
    print(f"[DEBUG] User Name: {user_name}")
    print(f"[DEBUG] User Input: {user_input}")

    system_prompt = f"""
    You are an expert Manager Agent. Your task is to analyze the conversation summary, think like a human to decompose the problem, and formulate an abstract plan to retrieve or store information. You must NOT answer the user directly.

    You have access to the following tools:
    1. GraphitiRead: A knowledge graph. Use it to search for structural relationships and entities related to the user.
    2. ChromaDBRead: A vector database. Use it to search for unstructured documents or manuals.
    3. GraphitiWrite: Use it to store important user details, preferences, or facts from the conversation into their personal memory graph.

    You must STRICTLY follow this two-step output format:

    Thought: <Write your step-by-step reasoning based on the summary. Explain what you need to search, what you need to store, or if it is just casual chit-chat> [2].
    Abstract Chain: <Generate the exact tool calls using the format [query -ToolName-> variable]. Use variables like y1, y2. If no tools are needed, write "None"> [3].

    CRITICAL RULES:
    - The format inside the brackets MUST be: [what to search or store -ToolName-> variable] [3].
    - Valid ToolNames are only: GraphitiRead, ChromaDBRead, GraphitiWrite.

    EXAMPLES:

    Summary: User (Juan) says he just bought a new car and wants to know how to change its oil.
    Thought: I need to store the fact that Juan bought a new car in his knowledge graph. Then, I need to search the vector database for the manual on how to change the oil.
    Abstract Chain: Store [Juan bought a new car -GraphitiWrite-> y1]. Search [how to change oil -ChromaDBRead-> y2].

    Summary: User (Maria) says "Good morning! How are you?"
    Thought: The user is just exchanging pleasantries. There is no new factual information to store and no specific question to research. I should just greet her back.
    Abstract Chain: None.

    User: Who is the manager of the Alpha Project and what are their main skills?
    Thought: I need to find the manager of the Alpha Project using the knowledge graph. Once I have the manager's name, I need to search the vector database for documents containing their skills.
    Abstract Chain: First, find the [manager of Alpha Project -GraphitiRead-> y1]. Then, find the [skills of y1 -ChromaDBRead-> y2].

    User: Me gusta mucho la pizza de piña.
    Thought: The user shared a personal preference about food. I need to store this fact in the knowledge graph memory so I can remember it for future interactions.
    Abstract Chain: Store [User likes pineapple pizza -GraphitiWrite-> y1].

    User: What is the operating temperature of the bolometer mentioned in the X-200 manual?
    Thought: I need to search the vector database for the X-200 manual to find the section about the bolometer, and then extract its operating temperature. 
    Abstract Chain: Search for [bolometer operating temperature in X-200 manual -ChromaDBRead-> y1].

    User: Where is the headquarters of Ralph Hefferline's company?
    Thought: I first need to find the company Ralph Hefferline works for in the knowledge graph. Then, I need to find the headquarters location of that specific company in the vector database.
    Abstract Chain: Find the [company of Ralph Hefferline -GraphitiRead-> y1]. Then, search for the [headquarters of y1 -ChromaDBRead-> y2].

    User: I am working on a new React component called 'MapViewer'.
    Thought: The user shared an update about their current work. I need to store this project information in the knowledge graph to keep track of their context.
    Abstract Chain: Store [User is working on a React component called MapViewer -GraphitiWrite-> y1].

    User: Hello, how are you?
    Thought: The user is just exchanging pleasantries. There is no factual information to store and no specific question to research. I just need to answer directly.
    Abstract Chain: None

    User: ¿Qué le gusta a Jorge?
    Thought: I need to check the knowledge graph to see what preferences or interests are stored for the entity Jorge.
    Abstract Chain: Search for [Jorge's preferences -GraphitiRead-> y1].


    Now, process the following user {user_name} query:
    {chat_summary}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    
    response = await llm_router.ainvoke(messages)
    content = response.content

    # Extract reasoning from Thought: or <think> tags
    reasoning = ""
    reasoning_match = re.search(r'(?:Thought:|<think>)\s*(.*?)(?=\nAbstract Chain:|</think>|$)', content, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Extract plan from Abstract Chain:
    abstract_plan = content
    plan_match = re.search(r'Abstract Chain:\s*(.*)', content, re.DOTALL | re.IGNORECASE)
    if plan_match:
        abstract_plan = plan_match.group(1).strip()

    print(f"[DEBUG] Planner Reasoning: {reasoning[:100]}...")
    print(f"[DEBUG] Abstract Plan: {abstract_plan}")

    return {
        "abstract_plan": abstract_plan,
        "reasoning": reasoning,
        "user_name": user_name,
        "chat_summary": chat_summary
    }

async def executor_node(state: AgentState, mcp_tools: list = None) -> dict:
    """
    Parses the abstract_plan and executes tool calls.
    Specifically for Graphiti MCP tools.
    """
    plan = state.get("abstract_plan", "")
    if "Abstract Chain: None" in plan or not plan:
        return {"steps": ["No plan to execute"], "tool_results": []}

    # Extract tool calls: [query -ToolName-> variable]
    tool_calls = re.findall(r'\[(.*?) -(GraphitiRead|ChromaDBRead|GraphitiWrite)-> (y\d+)\]', plan)
    
    if not tool_calls:
        print("[WARN] No tool calls found in the abstract plan.")
        return {"steps": ["Zero tool calls parsed"]}

    results = {}
    observations = []
    steps_taken = []

    print(f"[INFO] Executor found {len(tool_calls)} tool calls.")
    
    for query, tool_name, var_name in tool_calls:
        # Replace y1, y2 with previous results if they exist in the query
        for v_name, v_val in results.items():
            query = query.replace(v_name, str(v_val))
        
        print(f"[INFO] Executing: {query} with {tool_name} -> {var_name}")
        
        result = "Tool result placeholder"
        
        # Here we will eventually call the actual tool
        # If 'tools' is provided (e.g. via closure), we can use them.
        if mcp_tools:
            # Map new tool names to specific MCP tool functions
            tool_to_call = None
            if tool_name == "GraphitiRead":
                tool_to_call = next((t for t in mcp_tools if t.name == "search_nodes"), None)
            elif tool_name == "GraphitiWrite":
                tool_to_call = next((t for t in mcp_tools if t.name == "add_memory"), None)
            elif tool_name == "ChromaDBRead":
                tool_to_call = next((t for t in mcp_tools if t.name == "search_documents"), None)
            
            if tool_to_call:
                try:
                    # Execute using the detected user_name
                    user_id = state.get('user_name', 'usuario').lower()
                    
                    # Prepare arguments based on tool type
                    args = {"query": query, "group_id": user_id}
                    if tool_name == "GraphitiWrite":
                        args = {
                            "name": f"Interaction {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            "content": query,
                            "group_id": user_id,
                            "source": "message"
                        }
                    
                    tool_res = await tool_to_call.ainvoke(args)
                    result = tool_res
                except Exception as e:
                    result = f"Error calling {tool_name}: {e}"
        
        results[var_name] = result
        observations.append({
            "tool": tool_name,
            "query": query,
            "result": result,
            "variable": var_name
        })
        steps_taken.append(f"Called {tool_name} for '{query}' and stored in {var_name}")

    print(f"[INFO] Tool result: {observations}")

    return {
        "steps": steps_taken,
        "tool_results": observations,
        "observations": json.dumps(observations, indent=2, ensure_ascii=False)
    }

async def synthesizer(state: AgentState) -> dict:
    """ 
        This node synthesizes the final answer based on the current state of the agent.
    """

    tool_observations = state.get("observations", "")
    user_input = state.get("user_input", "")
    
    system_prompt = """
    You are a helpful and conversational AI assistant. Your task is to provide the final response to the user.

    You have been provided with the following background information and executed tool results:
    {tool_observations}
    (Note: If the observation says a fact was stored successfully, acknowledge it naturally).

    User's original query / intent: 
    {user_input}

    INSTRUCTIONS:
    Using the background information provided, formulate a natural, human-like response to the user. Do not mention "tool observations" or "variables" in your response. Just answer the query directly or continue the conversation smoothly.
    """

    messages = [
        ("system", system_prompt.format(tool_observations=tool_observations, user_input=user_input)),
        ("human", user_input)
    ]

    response = await llm_router.ainvoke(messages)
    content = response.content

    print(f"[INFO] Synthesizer response: {content}")

    return {
        "final_answer": content
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
     6. CRITICAL: Never include your internal reasoning, thinking process, or "Thinking Process" headers in your response. Output ONLY the response to the user.
    CURRENT TIME: {current_time}
    """

    # Collect ALL context from tool-related messages since the last HumanMessage
    tool_contexts = []
    
    # 1. Check for observations from our new executor
    if state.get("observations"):
        tool_contexts.append(f"Tool Observations: {state['observations']}")
    
    # 2. Check for tool results list
    if state.get("tool_results"):
        results_str = "\n".join([f"- {r['tool']}({r['query']}): {r['result']}" for r in state['tool_results']])
        tool_contexts.append(f"Results Summary:\n{results_str}")

    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, AIMessage) and any(x in m.content for x in ["Knowledge Base Info", "Weather Info", "Web Content Info"]):
            tool_contexts.append(m.content)
        # Also handle ToolMessages if they exist (from MCP tools directly)
        elif hasattr(m, 'tool_call_id'): # Generic check for tool message
             tool_contexts.append(f"Tool Result: {m.content}")

    if tool_contexts:
        prompt += "\n\nContext from tools:"
        for ctx in reversed(tool_contexts):
            prompt += f"\n- {ctx}"
    else:
        # If the user is just saying hi or stating something without tool results
        prompt += "\n\nNote: The user is making a statement or sharing information. Acknowledge naturally. No specific tool results found."
    
    # print(f"[DEBUG] Final Answer Prompt: {prompt}")

    messages = [SystemMessage(content=prompt)]
    
    # Add conversation history (excluding the meta-messages we just synthesized into the prompt)
    for m in state["messages"]:
        if isinstance(m, AIMessage) and any(x in m.content for x in ["Knowledge Base Info", "Weather Info", "Web Content Info"]):
            continue
        messages.append(m)

    response = await llm_text.ainvoke(messages)
    
    # 1. Extract reasoning before stripping
    reasoning = ""
    # Standard XML-like tags (and variations)
    reasoning_match = re.search(r'<(think|reasoning|thought)>(.*?)</\1>', response.content, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(2).strip()
    
    # 2. Aggressive Reasoning Stripper
    # We strip any typical markers and anything that looks like internal deliberation.
    
    # Standard XML tags (DeepSeek style)
    response.content = re.sub(r'<(think|reasoning|thought|deliberation)>.*?</\1>', '', response.content, flags=re.DOTALL | re.IGNORECASE).strip()
    
    # Common headers and their typical follow-up numbering
    # We strip from the header until the end of the line, and if it's at the start, we try to find the actual answer
    
    # If the response contains "Thinking Process" or similar, we take everything AFTER the last occurrence 
    # if there is a clear separation, OR we strip the blocks.
    
    if "Thinking Process" in response.content or "Thought:" in response.content or "Reasoning:" in response.content:
        # Heuristic: The actual answer is usually the last paragraph or the part that doesn't look like reasoning
        # Let's split by double newline and filter
        parts = re.split(r'\n\n|\n(?=[A-Z])', response.content)
        filtered_parts = []
        for p in parts:
            p_strip = p.strip()
            # If the block mentions "thinking process", "analyze", "goal" or starts with numbers/bullets, skip it
            is_noise = re.match(r'^(Thinking Process|Reasoning|Thought|Analysis|Rationale|Summary|Step \d|[\d\.]+\s+[A-Z]|\*|\-)', p_strip, re.IGNORECASE)
            # Also skip if it seems too "meta"
            if is_noise or any(x in p_strip.lower() for x in ["determine the goal", "analyze the request", "evaluate constraints"]):
                continue
            
            filtered_parts.append(p_strip)
        
        if filtered_parts:
            response.content = "\n\n".join(filtered_parts).strip()
    
    # Final cleanup of common leftover prefixes
    headers_to_strip = [
        r'^MapirBot:\s*',
        r'^Final Answer:\s*',
        r'^Respuesta:\s*',
        r'^Entendido\.\s*', # If the model repeats it
    ]
    for pattern in headers_to_strip:
        response.content = re.sub(pattern, '', response.content, flags=re.IGNORECASE).strip()

    if not response.content:
        response.content = "Entendido." # Ultra-dry fallback

    return {
        "messages": [response], 
        "next_step": None, 
        "reasoning": reasoning
    }

async def log_interaction(state: AgentState, tools: list = None) -> dict:
    """
    Background node that logs the interaction to Graphiti.
    It summarizes the user statement and agent response.
    """
    if not tools:
        return {}

    # ONLY use the absolute latest HumanMessage and the latest AIMessage for logging
    # to avoid context overflow (400 - Context size exceeded)
    user_msg = ""
    bot_msg = ""
    
    # Traverse in reverse to find the very last exchange
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and not bot_msg:
            bot_msg = m.content
        elif isinstance(m, HumanMessage) and not user_msg:
            user_msg = m.content
        
        if user_msg and bot_msg:
            break

    if not user_msg:
        return {}

    user_name = state.get("user_name", "Usuario")
    user_input = state.get("user_input", "")

    # Define background worker
    async def worker():
        try:
            # Aggressively trim strings to save context
            u_short = user_msg[:400] 
            summary_prompt = f"""
            Extract the essence of what ONLY the USER said. Ignore the bot's response.
            User: {user_name}
            User shared: "{u_short}"
            
            Synthesize into a DRY statement.
            Example: "Jorge: le gusta el pádel." or "Jorge: está trabajando en un componente React."
            
            Format: "{user_name}: <fact>"
            Return ONLY the formatted string or 'NONE' if trivial.
            """
            
            # Use a fresh, stateless call to minimize overhead
            summary_res = await llm_text.ainvoke(summary_prompt)
            summary = summary_res.content.strip()

            # Clean up potential thinking or markdown
            summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL | re.IGNORECASE).strip()
            summary = summary.replace('"', '').replace("'", "")

            if "NONE" in summary.upper() or len(summary) < 5:
                return

            # Call add_memory tool
            add_memory_tool = next((t for t in tools if t.name == "add_memory"), None)
            if add_memory_tool:
                # Standardize name for Graphiti group_id
                group_id = user_name.lower()
                print(f"[INFO] Background Logging for {user_name}: {summary}")
                await add_memory_tool.ainvoke({
                    "name": f"Interaction {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "content": summary,
                    "group_id": group_id,
                    "source": "message"
                })
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[ERROR] Background logging failed for {user_name}:\n{error_details}")

    # Launch task and return immediately
    asyncio.create_task(worker())

    return {"steps": [f"Interaction logging started for {user_name}"]}

async def trim_memory(state: AgentState) -> dict:
    """ 
        This node trims the message history to keep only the last N messages.
    """
    messages = state.get("messages", [])
    window_size = 6 # Keep 3 turns
    
    if len(messages) > window_size:
        # We only return the RemoveMessage list to LangGraph
        return {
            "messages": [RemoveMessage(id=m.id) for m in messages[:-window_size]]
        }
            
    return {"messages": []}