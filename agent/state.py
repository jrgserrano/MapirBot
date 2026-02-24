from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # History of messages
    summary: Optional[str] # Summary of the conversation
    current_research: Optional[str] # Current research topic
    user_context: Optional[str] # User context
    next_step: Optional[str] # Next step to take
    current_time: Optional[str] # ISO format current time