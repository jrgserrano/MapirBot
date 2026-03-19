import asyncio
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from agent.state import AgentState
from agent.nodes import worker, planner, executor_node, synthesizer 
from agent.llm import db_checkpointer, llm_main

def create_graph(checkpointer, tools=None):
    workflow = StateGraph(AgentState)
    
    async def executor(state: AgentState):
        return await executor_node(state, mcp_tools=tools)

    #async def logger(state: AgentState):
    #    return await log_interaction(state, tools=tools)
    
    workflow.add_node("worker", worker)
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("synthesizer", synthesizer)
    #workflow.add_node("log_interaction", logger)
    #workflow.add_node("trim_memory", trim_memory)
    
    # Set Entry Point
    workflow.set_entry_point("worker")
    
    # Transitions
    workflow.add_edge("worker", "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "synthesizer")
    workflow.add_edge("synthesizer", END)
    #workflow.add_edge("final_answer", "log_interaction")
    #workflow.add_edge("log_interaction", "trim_memory")
    #workflow.add_edge("trim_memory", END)
    
    return workflow.compile(checkpointer=checkpointer)