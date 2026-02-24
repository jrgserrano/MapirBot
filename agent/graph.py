import asyncio
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from agent.state import AgentState
from agent.nodes import supervisor_agent, final_answer_agent, knowledge_node, trim_memory, weather_node, web_scraper_node, memory_node
from agent.llm import db_checkpointer, llm_main

def create_graph(tools, checkpointer):
    tools_node = ToolNode(tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("trim_memory", trim_memory)
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("final_answer", final_answer_agent)
    workflow.add_node("knowledge_base", knowledge_node)
    workflow.add_node("weather", weather_node)
    workflow.add_node("web_scraper", web_scraper_node)
    workflow.add_node("paper_search_tools", tools_node)

    async def router(state: AgentState) -> str:
        next_step = state.get("next_step", "WRITE")
        if not next_step:
            return "supervisor"
        
        next_step = next_step.strip().upper()
        if "SEARCH_KNOWLEDGE" in next_step:
            return "knowledge_base"
        elif "SEARCH_WEATHER" in next_step:
            return "weather"
        elif "SEARCH_WEB_PAPERS" in next_step:
            return "paper_search_agent"
        elif "SCRAPE_URL" in next_step:
            return "web_scraper"
        elif "MEMORY_UPDATE" in next_step:
            return "memory_node"
        elif "WRITE" in next_step:
            return "final_answer"
        return "supervisor"

    async def paper_search_agent(state: AgentState):
        """ This node uses an LLM bound with MCP tools to generate tool calls """
        from agent.llm import llm_text
        llm_with_tools = llm_text.bind_tools(tools)
        prompt = "You are a research assistant. Use the available tools to find papers or technical info about the user's query."
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    workflow.add_node("paper_search_agent", paper_search_agent)
    workflow.add_node("memory_node", memory_node)

    workflow.set_entry_point("trim_memory")
    workflow.add_edge("trim_memory", "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        router,
        {
            "knowledge_base": "knowledge_base",
            "weather": "weather",
            "paper_search_agent": "paper_search_agent",
            "web_scraper": "web_scraper",
            "memory_node": "memory_node",
            "final_answer": "final_answer",
            "supervisor": "supervisor"
        }
    )
    workflow.add_edge("knowledge_base", "final_answer")
    workflow.add_edge("weather", "final_answer")
    workflow.add_edge("web_scraper", "final_answer")
    workflow.add_edge("memory_node", "final_answer")
    
    # Paper search flow
    workflow.add_edge("paper_search_agent", "paper_search_tools")
    workflow.add_edge("paper_search_tools", "final_answer")

    workflow.add_edge("final_answer", END)
    
    return workflow.compile(checkpointer=checkpointer)