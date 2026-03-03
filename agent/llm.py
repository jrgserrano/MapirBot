from langchain_ollama import ChatOllama
from tools.knowledge_base import knowledge_base, knowledge_base_update

# Checkpointer will be initialized asynchronously in the main app
db_checkpointer = None

# Main LLM for text generation
llm_text = ChatOllama(
    model="gpt-oss:20b", # "qwen3-vl:8b",
    num_ctx=8192,
    # num_thread=8,
    temperature=0.7
)

# Targeted, fast router for graph logic (llama3.2:3b is much faster for routing)
llm_router = ChatOllama(
    model="gpt-oss:20b",
    num_ctx=8192,
    # num_thread=8,
    temperature=0
)

# LLM bound to tools for technical searches
llm_main = llm_text.bind_tools([knowledge_base, knowledge_base_update])