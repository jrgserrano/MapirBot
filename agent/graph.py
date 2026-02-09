from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import supervisor_agent, final_answer_agent, knowledge_node, trim_memory
from agent.llm import db_checkpointer

def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("trim_memory", trim_memory)
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("final_answer", final_answer_agent)
    workflow.add_node("knowledge_base", knowledge_node)

    def router(state: AgentState) -> str:
        next_step = state["next_step"].strip().upper()
        if "SEARCH_KNOWLEDGE" in next_step or "RESEARCH" in next_step:
            return "knowledge_base"
        elif "WRITE" in next_step:
            return "final_answer"
        return "supervisor"
    
    workflow.set_entry_point("trim_memory")
    workflow.add_edge("trim_memory", "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        router,
        {
            "knowledge_base": "knowledge_base",
            "final_answer": "final_answer",
            "supervisor": "supervisor"
        }
    )
    workflow.add_edge("knowledge_base", "supervisor")
    workflow.add_edge("final_answer", END)
    return workflow.compile(checkpointer=db_checkpointer)

graph_app = create_graph()