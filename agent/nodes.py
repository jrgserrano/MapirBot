from langchain_core.messages import SystemMessage
from agent.llm import llm_main
from agent.state import AgentState
from langchain_core.messages import RemoveMessage, HumanMessage, AIMessage
from tools.knowledge_base import knowledge_base

def supervisor_agent(state: AgentState) -> dict:
    """ This node decides what action to take based on the current state of the agent """

    prompt = f"""
    You are the "MapirBot Router". Your only job is to direct the flow of a conversation 
    between a human and a technical assistant specialized in Robotics and C++.

    CONTEXT SUMMARY:
    {state.get('summary', 'No previous context.')}

    DECISION RULES:
    1. If the user's message is a greeting (like "Hola", "Hello") or small talk: 
       -> Return 'WRITE' immediately.
    2. If the user asks a technical question or for code: 
       -> Return 'SEARCH_KNOWLEDGE'.
    3. If the information needed is already present in the SUMMARY: 
       -> Return 'WRITE'.

    ALLOWED ACTIONS:
    - SEARCH_KNOWLEDGE: Use this to search the technical database.
    - WRITE: Generate the final direct response.

    RESPONSE FORMAT:
    Return only 'SEARCH_KNOWLEDGE' or 'WRITE'. No punctuation.
"""

    messages = [ SystemMessage(content=prompt)] + state["messages"]

    response = llm_main.invoke(messages)

    return {"next_step": response.content}

def knowledge_node(state: AgentState) -> dict:
    """ This node retrieves and synthesizes information from the knowledge base """

    query = state["messages"][-1].content
    search_results = knowledge_base.invoke(query)
    
    # Synthesis Prompt: Extract only the core facts
    synthesis_prompt = f"""
    SYSTEM: You are a technical data extractor.
    GOAL: Extract ONLY the technical facts relevant to the user's query from the search results below.
    If no relevant info is found, say "No relevant info found in the database".
    
    USER QUERY: {query}
    SEARCH RESULTS: {search_results}
    
    FORMAT: Bullet points, extremely concise.
    """
    
    synthesized_info = llm_main.invoke(synthesis_prompt)
    
    return {"messages": [AIMessage(content=f"Knowledge Base Info:\n{synthesized_info.content}")], "next_step": "SUPERVISOR"}
    
def final_answer_agent(state: AgentState) -> dict:
    """ This node writes the final answer based on the current state of the agent """

    prompt = f"""
    ROLE: 
    You are MapirBot, a senior robotics and C++ engineer. 
    You are a teammate in this Slack channel. 

    TONE & STYLE (CRITICAL):
    - BE HUMAN: No "As an AI...", no helpful bot clichés.
    - ULTRA-CONCISE: 1-2 sentences max. Don't explain your process. Just answer.
    - DIRECT: No greetings ("Hi Jorge"), no sign-offs ("Hope this helps").
    - EXPERT: Use technical terms correctly. If you don't know, say "no clue" or "not in the docs".
    - STYLE: Lowercase is fine. Sarcasm/dry wit is encouraged. Act like you're busy coding.

    CONTEXT:
    {state.get('summary', 'No previous context.')}

    GOAL:
    Give the user the absolute minimum info needed to move forward.
    """

    prompt += f"\n\nContext from tools: {state['messages'][-1].content}"

    messages = [ SystemMessage(content=prompt)] + state["messages"]

    response = llm_main.invoke(messages)

    return {"messages": [response], "next_step": None} # Clear next_step after use

def trim_memory(state: AgentState) -> dict:
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

            updated_summary = llm_main.invoke(prompt)
            
            return {"messages": [RemoveMessage(id=m.id) for m in messages[:to_remove]], "summary": updated_summary.content}
            
    return {"messages": [], "summary": prev_summary}