import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from tools.knowledge_base import knowledge_base, knowledge_base_update

conn = sqlite3.connect("database/checkpoints.sqlite", check_same_thread=False)
db_checkpointer = SqliteSaver(conn)

# Supervisor LLM
llm_main = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="not-needed",
    model_name="local-model",
    temperature=0.7
    )
#.bind_tools([knowledge_base, knowledge_base_update])