from langchain_ollama import ChatOllama
from tools.knowledge_base import knowledge_base, knowledge_base_update

# Checkpointer will be initialized asynchronously in the main app
db_checkpointer = None

# Main LLM for text generation (unbound)
llm_text = ChatOllama(
    model="llama3.2:3b", #"qwen3-vl:8b",
    num_ctx=8192,
    num_thread=8,
    temperature=0.3
)

# LLM bound to tools for technical searches
llm_main = llm_text.bind_tools([knowledge_base, knowledge_base_update])

# Targeted, fast router for graph logic (llama3.2:3b is much faster for routing)
llm_router = ChatOllama(
    model="llama3.2:3b",
    num_ctx=8192,
    num_thread=8,
    temperature=0
)