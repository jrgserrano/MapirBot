from langchain_openai import ChatOpenAI
from tools.knowledge_base import knowledge_base, knowledge_base_update

# Checkpointer will be initialized asynchronously in the main app
db_checkpointer = None

# LM Studio uses an OpenAI-compatible API on port 1234 by default
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# Main LLM for text generation
llm_text = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="lm-studio", # Any string works for LM Studio
    model="mistral-nemo-12b-instruct-2407", # Name must match the one loaded in LM Studio
    temperature=0.7
)

# Targeted, fast router for graph logic
llm_router = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key="lm-studio",
    model="mistral-nemo-12b-instruct-2407",
    temperature=0
)

# LLM bound to tools for technical searches
llm_main = llm_text #.bind_tools([knowledge_base, knowledge_base_update])