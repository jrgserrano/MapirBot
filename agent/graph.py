import asyncio
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.client import MultiServerMCPClient
from agent.state import AgentState
from agent.nodes import planner
from agent.llm import db_checkpointer, llm_main

def create_graph(checkpointer):
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner)
    
    workflow.set_entry_point("planner")

    # End node
    workflow.add_edge("planner", END)
    
    return workflow.compile(checkpointer=checkpointer)