from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from datetime import datetime
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # History of messages
    summary: Optional[str] # Summary of the conversation
    current_research: Optional[str] # Current research topic
    user_context: Optional[str] # User context
    next_step: Optional[str] # Next step to take
    current_time: Optional[str] = lambda: datetime.now().isoformat() # ISO format current time
    logs: Optional[List[str]] # Feedback logs for the user
    loop_count: Annotated[int, operator.add] # Counter for loop control
    evaluator_feedback: Optional[str] # Internal feedback from evaluator
    is_valid: Optional[bool] # Evaluator flag if the answer is completely valid