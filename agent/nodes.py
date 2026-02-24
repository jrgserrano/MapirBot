from langchain_core.messages import SystemMessage
from agent.llm import llm_main, llm_router, llm_text
from agent.state import AgentState
from langchain_core.messages import RemoveMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import Literal
import re
import json
from tools.knowledge_base import knowledge_base, knowledge_base_update
from tools.weather import get_weather
from tools.web_scraper import scrape_web
from agent.graphiti_client import get_graphiti
from graphiti_core.nodes import EpisodeType
from datetime import datetime, timezone

class Router(BaseModel):
    """Call this tool to route the conversation to the next specialized node."""
    next_step: Literal["SEARCH_KNOWLEDGE", "SEARCH_WEATHER", "SEARCH_WEB_PAPERS", "SCRAPE_URL", "WRITE"] = Field(
        description="The next action to take."
    )

async def supervisor_agent(state: AgentState) -> dict:
    """ This node decides what action to take using structured output and sequential logic """

    current_time = state.get("current_time", "Unknown")

    prompt = f"""
    You are the "MapirBot Supervisor", an expert at orchestrating technical information retrieval.
    Your goal is to answer the user's query by delegating to the right specialized tools.
    CURRENT TIME (UTC): {current_time}

    --- TOOLS AVAILABLE ---
    - SEARCH_KNOWLEDGE: Hybrid database using Vector Storage and a Relational Knowledge Graph. 
      Use this for deep context about projects, team members, or technical manuals.
    - SEARCH_WEATHER: Precise weather data for a location.
    - SEARCH_WEB_PAPERS: Deep technical research using scholarly sources (MCP tools). Use this if SEARCH_KNOWLEDGE is not enough.
    - SCRAPE_URL: Extracts information directly from a specific link/URL provided by the user.
    - MEMORY_UPDATE: Call this if the user provides personal information, preferences, or project details that should be remembered in the long-term knowledge graph. 
    - WRITE: Send the final response to the user.

    Answer ONLY with the JSON representing the next step.
    """

    try:
        structured_llm = llm_router.with_structured_output(Router)
        messages = [SystemMessage(content=prompt)] + state["messages"]
        decision = await structured_llm.ainvoke(messages)
        
        print(f"[DEBUG] Supervisor decision: {decision}")
        
        return {"next_step": decision.next_step}
    except Exception as e:
        print(f"[ERROR] Supervisor structured output failed: {e}")
        return {"next_step": "WRITE"}

async def web_scraper_node(state: AgentState) -> dict:
    """ This node extracts content from a URL """
    last_message = state["messages"][-1].content
    url_match = re.search(r"(https?://[^\s]+)", last_message)
    
    if not url_match:
        return {"messages": [AIMessage(content="Web Content Info: No URL found in request.")], "next_step": "SUPERVISOR"}
        
    url = url_match.group(0)
    print(f"[INFO] Scraping URL: {url}...")
    content = await scrape_web.ainvoke(url)
    
    return {"messages": [AIMessage(content=f"Web Content Info from {url}:\n{content}")], "next_step": "SUPERVISOR"}

async def weather_node(state: AgentState) -> dict:
    """ This node retrieves weather information accurately and quickly """

    query = state["messages"][-1].content
    
    # Pass through LLM for extraction
    extraction_prompt = f"Extract city from: '{query}'. If none, say 'Malaga'. Result: [CITY]"
    location_raw = (await llm_router.ainvoke(extraction_prompt)).content.strip()
    
    # Strip everything but the word
    location = re.sub(r'<think>.*?</think>', '', location_raw, flags=re.DOTALL)
    # Extract the last word or the capitalized one
    words = re.findall(r'[A-Z][a-z]+', location)
    if words:
        location = words[-1]
    else:
        location = "Malaga"
    
    weather_info = await get_weather.ainvoke(location)
    
    return {"messages": [AIMessage(content=f"Weather Info for {location}:\n{weather_info}")], "next_step": "SUPERVISOR"}

async def knowledge_node(state: AgentState) -> dict:
    """ This node retrieves and synthesizes information from the knowledge base """

    query = state["messages"][-1].content
    search_results = await knowledge_base.ainvoke(query)
    current_time = state.get("current_time", "Unknown")
    
    # Synthesis Prompt: Extract only the core facts
    synthesis_prompt = f"""
    SYSTEM: You are a technical data extractor.
    GOAL: Extract ONLY the technical facts relevant to the user's query from the search results below.
    If no relevant info is found, say "No relevant info found in the database".
    
    CURRENT TIME (UTC): {current_time}
    USER QUERY: {query}
    SEARCH RESULTS: {search_results}
    
    FORMAT: Bullet points, extremely concise. 
    NOTE: If there are temporal references in the search results, use the current time to reason about them (e.g., "Yesterday" refers to the day before {current_time}).
    """
    
    synthesized_info = await llm_main.ainvoke(synthesis_prompt)
    
    return {"messages": [AIMessage(content=f"Knowledge Base Info:\n{synthesized_info.content}")], "next_step": "SUPERVISOR"}
    
async def final_answer_agent(state: AgentState) -> dict:
    """ This node writes the final answer based on the current state of the agent """

    current_time = state.get("current_time", "Unknown")

    prompt = f"""
    ROLE: MapirBot, a senior roboti
    cs engineer. You are a colleague on Slack.
    CURRENT TIME: {current_time}

    STRICT GUIDELINES:
    1. CONCISE: Max 1-2 sentences. 
    2. NATURAL: No "As an AI", "According to database", or "I don't have access".
    3. DIRECT: Give facts directly (e.g., "Alice is in Malaga" instead of "The tool says Alice is in Malaga").
    4. MATCH LANGUAGE: Respond in the same language as the user.
    """

    prompt += "\n\nContext from tools and history:"
    
    # Collect ALL context from tool-related messages since the last HumanMessage
    tool_contexts = []
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, AIMessage) and any(x in m.content for x in ["Knowledge Base Info", "Weather Info", "Web Content Info", "Paper research results"]):
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
        if isinstance(m, AIMessage) and any(x in m.content for x in ["Knowledge Base Info", "Weather Info", "Web Content Info", "Paper research results"]):
            continue
        messages.append(m)

    response = await llm_text.ainvoke(messages)
    
    # Strip reasoning if present
    response.content = re.sub(r'<think>.*?</think>', '', response.content, flags=re.DOTALL).strip()
    
    if response.content:
        print(f"[INFO] Final Answer: {response.content[:100]}...")
    else:
        print("[WARN] Final Answer is EMPTY!")
        response.content = "No he podido procesar una respuesta clara con la información obtenida. ¿Podrías reformular la pregunta?"

    return {"messages": [response], "next_step": None} # Clear next_step after use


async def memory_node(state: AgentState) -> dict:
    """ This node extracts and stores memorable info from prompts into Graphiti """
    
    last_human_message = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_human_message = m.content
            break
            
    if not last_human_message:
        return {"next_step": "SUPERVISOR"}

    current_time = state.get("current_time", datetime.now(timezone.utc).isoformat())

    # Reasoning prompt to decide if information is relevant for the graph
    check_prompt = f"""
    SYSTEM: You are a "Memory Filter" for a MapirBot agent. 
    Your job is to decide if the user's message contains new, relevant, or personal information that should be remembered in a long-term Knowledge Graph.
    
    FACTS TO REMEMBER:
    - User's name, job, location, preferences.
    - Relationships between people or entities.
    - Specific project details or roles mentioned by the user.
    - Direct statements of fact ("I am...", "Alice works at...", "My favorite tool is...").
    
    FACTS TO IGNORE:
    - Greetings ("Hola", "Hi").
    - Questions about the weather/time/papers.
    - General coding help requests unless they reveal something about the user.
    - Temporary task-specific info.

    DECISION CRITERIA:
    - Is this information already known? (Assume it's NOT known if you aren't sure).
    - Is it important enough for a colleague to remember later?
    
    USER MESSAGE: "{last_human_message}"

    Respond ONLY with a JSON object:
    {{
        "is_relevant": true/false,
        "reason": "short explanation",
        "fact_to_save": "The concise fact to store in the graph (in English)"
    }}
    """

    try:
        decision_raw = await llm_router.ainvoke(check_prompt)
        # Clean potential reasoning
        content = re.sub(r'<think>.*?</think>', '', decision_raw.content, flags=re.DOTALL).strip()
        decision = json.loads(content)
        
        if decision.get("is_relevant"):
            print(f"[INFO] Memory Node: Found relevant info - {decision['reason']}")
            
            graphiti = get_graphiti()
            await graphiti.add_episode(
                name="User Prompt Insight",
                episode_body=decision["fact_to_save"],
                source_description="Extracted from user prompt",
                source=EpisodeType.text,
                reference_time=datetime.fromisoformat(current_time) if isinstance(current_time, str) else current_time
            )
            print(f"[INFO] Memory Node: Episode added to Graphiti: {decision['fact_to_save']}")
        else:
            print(f"[INFO] Memory Node: Ignoring message (not relevant for long-term memory).")
            
    except Exception as e:
        print(f"[ERROR] Memory node failed: {e}")

    return {"next_step": "SUPERVISOR"}

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