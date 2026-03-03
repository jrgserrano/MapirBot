# MapirBot

MapirBot is an advanced autonomous AI assistant built with **LangGraph**, **LangChain**, and **Ollama/OpenAI**. It features a hybrid memory architecture, MCP tool integration, autonomous **WebRover-style** web search, and a supervisor-led workflow.

## Project Structure

```text
.
├── agent/
│   ├── graph.py          # Workflow orchestration and routing using LangGraph.
│   ├── nodes.py          # Node logic (Supervisor, Memory Node, Final Answer, etc.).
│   ├── state.py          # State definition (messages, summary, metadata).
│   ├── llm.py            # LLM initialization (Main, Router, Text) and persistence.
│   └── graphiti_client.py # Singleton client for Graphiti (Neo4j Knowledge Graph).
├── tools/
│   ├── knowledge_base.py # Hybrid Search (Chroma Vector DB + Graphiti Graph).
│   ├── weather.py        # Weather retrieval tool.
│   └── web_scraper.py    # URL content extraction tool.
├── database/             # SQLite checkpoints and ChromaDB persistence.
├── server.py             # FastAPI server with MCP client lifecycle management.
├── .env                  # Environment variables (Neo4j, LLM, Slack).
└── requirements.txt      # Python dependencies.
```

## Key Features

- **Autonomous Web Search (WebRover Style)**: Utilizes DuckDuckGo for independent link discovery and multi-step research.
- **Hybrid Memory Architecture**:
    - **Vector Storage (Chroma)**: Long-term storage for documents and technical manuals.
    - **Knowledge Graph (Graphiti/Neo4j)**: Persistent storage for entities, relationships, and user habits extracted from prompts.
- **Supervisor Workflow**: A central node that intelligently routes tasks to specialized nodes based on user intent.
- **Episodic Memory Extraction**: A dedicated `memory_node` identifies and saves relevant personal info or project details into the Knowledge Graph during conversation.
- **MCP Integration**: Uses the Model Context Protocol (MCP) to connect with external research tools (e.g., Paper Search).
- **Dynamic Context Management**: Automatic sliding window for message history with contextual summarization to preserve long-term relevance.
- **Persistence**: Full conversation state persistence using SQLite checkpoints.

## Prerequisites

1.  **Neo4j**: Required for the Knowledge Graph (Graphiti).
2.  **Ollama** (or OpenAI-compatible API): Running local models (e.g., `llama3.2`, `mxbai-embed-large`).
3.  **Environment Variables**:
    ```bash
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password
    OPENAI_BASE_URL=http://localhost:11434/v1
    MODEL_NAME=llama3.2
    ```

## Installation

1.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Try It

To run the end-to-end test and see the agent in action:

```bash
python3 scripts/test_graph.py
```

## Architecture Overview

The agent is orchestrated using **LangGraph**. The workflow follows this path:

1.  **`trim_memory`**: Trims history and updates the conversation summary.
2.  **`supervisor`**: Analyzes the query and decides the next step:
    - `SEARCH_KNOWLEDGE` → `knowledge_base` (Chroma + Graphiti)
    - `SEARCH_WEATHER` → `weather_node`
    - `SEARCH_WEB_PAPERS` → `paper_search_agent` (MCP)
    - `SEARCH_WEB_ROVER` → `web_research` (Autonomous multi-step search)
    - `SCRAPE_URL` → `web_scraper_node`
    - `MEMORY_UPDATE` → `memory_node` (Enrich Knowledge Graph)
    - `WRITE` → `final_answer`
3.  **Synthesizer**: Tools return information for final short and clear response generation.
