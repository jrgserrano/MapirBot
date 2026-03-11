from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from datetime import datetime
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # History of messages
    query: Optional[str]
    inputs: Optional[str]
    abstract_plan: Optional[str]
    observations: Optional[str]
    final_answer: Optional[str]
    