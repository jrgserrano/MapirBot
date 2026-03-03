import asyncio
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from agent.state import AgentState
from agent.nodes import supervisor_agent, final_answer_agent, knowledge_node, trim_memory, weather_node, web_scraper_node, memory_node, web_research_node
from agent.llm import db_checkpointer, llm_main

def create_graph(checkpointer):
    workflow = StateGraph(AgentState)
    workflow.add_node("trim_memory", trim_memory)
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("final_answer", final_answer_agent)
    workflow.add_node("knowledge_base", knowledge_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("web_scraper", web_scraper_node)
    workflow.add_node("web_research", web_research_node)

    async def router(state: AgentState) -> str:
        # Loop Control Mechanism
        if state.get("loop_count", 0) >= 4:
            return "final_answer" # Force exit resolving with current context
            
        next_step = state.get("next_step", "WRITE")
        if not next_step:
            return "supervisor"
        
        next_step = next_step.strip().upper()
        if "SEARCH_KNOWLEDGE" in next_step:
            return "knowledge_base"
        elif "SEARCH_WEATHER" in next_step:
            return "weather"
        elif "SEARCH_WEB_ROVER" in next_step:
            return "web_research"
        elif "SCRAPE_URL" in next_step:
            return "web_scraper"
        elif "MEMORY_UPDATE" in next_step:
            return "memory_node"
        elif "WRITE" in next_step:
            return "final_answer"
        return "supervisor"

    workflow.add_node("memory_node", memory_node)

    workflow.set_entry_point("trim_memory")
    workflow.add_edge("trim_memory", "supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        router,
        {
            "knowledge_base": "knowledge_base",
            "weather": "weather",
            "web_scraper": "web_scraper",
            "web_research": "web_research",
            "memory_node": "memory_node",
            "final_answer": "final_answer",
            "supervisor": "supervisor"
        }
    )
    
    # Hub-and-Spoke Pattern: Tools return the control to the Supervisor!
    workflow.add_edge("knowledge_base", "supervisor")
    workflow.add_edge("weather", "supervisor")
    workflow.add_edge("web_scraper", "supervisor")
    workflow.add_edge("web_research", "supervisor")
    workflow.add_edge("memory_node", "supervisor")
    
    # End node
    workflow.add_edge("final_answer", END)
    
    return workflow.compile(checkpointer=checkpointer)