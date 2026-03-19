from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from datetime import datetime
import operator

class AgentState(TypedDict):
    user_input: Optional[str]
    chat_summary: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages] # History of messages
    abstract_plan: Optional[str]
    observations: Optional[str]
    final_answer: Optional[str]
    reasoning: Optional[str]
    steps: Annotated[List[str], operator.add]
    tool_results: Annotated[List[dict], operator.add]
    user_name: Optional[str]
    