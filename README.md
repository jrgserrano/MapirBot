# MapirBot

MapirBot is an autonomous AI assistant built with **LangGraph**, **LangChain**, and **OpenAI (LM Studio backend)**. It features memory management, persistence, and a supervisor-led workflow.

## Project Structure

```text
.
├── agent/
│   ├── graph.py          # Workflow orchestration using LangGraph.
│   ├── nodes.py          # Core logic for each step (Supervisor, Writer, Trimming).
│   ├── state.py          # State definition (messages, research, summary, etc.).
│   └── llm.py            # LLM initialization and persistence setup.
├── scripts/
│   └── test_graph.py     # End-to-end test script for the agent.
├── database/            # Contains SQLite persistence for session memory.
├── .env                 # Environment variables (Slack API keys, LLM URL).
└── requirements.txt     # Python dependencies.
```

## Key Features

- **Supervisor Workflow**: A central node decides whether the agent should research, think, or write the final answer.
- **Short-Term Memory**: Conversation history is persisted across restarts using a SQLite backend.
- **Dynamic Sliding Window**: To keep the context lean, the agent automatically trims history (keeps the last 10 messages).
- **Contextual Summarization**: When messages are trimmed, the agent automatically updates a conversation `summary` so it never truly "forgets" the context.
- **Role Continuity**: Ensures strict alternating roles (user/assistant) in the history to comply with LLM backend requirements.

## Prerequisites

1.  **LM Studio** (or another OpenAI-compatible local server) running at `http://127.0.0.1:1234/v1`.
2.  **Environment Variables**: Create a `.env` file in the root directory:
    ```bash
    OPENAI_API_KEY=not-needed
    LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
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

This script will:
1.  Introduce a user to the agent.
2.  Verify that the agent remembers the user's name across turns.
3.  Demonstrate the supervisor's decision-making process.

## Architecture Diagram

The workflow follows this logical path:
`trim_memory` (summarization) → `supervisor` (decision) → `final_answer` (response).
