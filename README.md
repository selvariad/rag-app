# RAG Enterprise Document Assistant

Enterprise document Q&A platform — upload documents, ask questions, get answers with source citations. Built on [rag-core](https://github.com).

## Features

- **Hybrid search** — ChromaDB vector search + BM25 keyword search + optional BGE-Reranker
- **Multi-provider LLM** — OpenAI, Anthropic, DeepSeek, Zhipu via LangChain
- **Self-correcting RAG** — LangGraph 4-node state machine with rewrite/retrieve/generate/check loop
- **Streaming SSE** — token-by-token response with real-time reasoning chain display
- **Source citations** — every answer cites the documents it used
- **Deduplication** — SHA-256 file hashing prevents duplicate ingestion
- **Conversation memory** — multi-turn chat with thread persistence
- **Dark/Light theme** — ChatGPT-inspired responsive UI
- **Standalone config editor** — `config.html` for API profile management (no server needed)

## Quick Start

```bash
# Install rag-core first
cd ../rag-core && pip install -e .

# Install rag-app
cd ../rag-app && pip install -e ".[dev]"

# Configure API keys
python -m rag_app.config_cli set llm.provider deepseek
python -m rag_app.config_cli set llm.api_key sk-your-key

# Or use the visual config editor
open config.html

# Start
python main.py
# → http://127.0.0.1:8000
```

## Configuration

Edit `config.yaml` or use the CLI:

```bash
python -m rag_app.config_cli show
python -m rag_app.config_cli set llm.provider openai
python -m rag_app.config_cli set llm.model gpt-4o-mini
python -m rag_app.config_cli set embedding.provider openai
```

Or use `config.html` — a standalone browser-based profile manager that can save multiple API configurations.

## Project Structure

```
rag-app/
├── main.py                  # Entry point
├── config.yaml              # Runtime configuration
├── config.html              # Standalone config editor
├── rag_app/
│   ├── config.py            # Config dataclasses + YAML loader
│   ├── deps.py              # Dependency injection (global singletons)
│   ├── graph.py             # LangGraph RAG state machine
│   ├── api.py               # FastAPI routes
│   ├── app.py               # App factory + lifespan
│   ├── tasks.py             # ARQ background tasks
│   ├── worker.py            # ARQ worker process
│   └── ui/
│       ├── routes.py        # Page routes
│       ├── templates/       # Jinja2 HTML templates
│       └── static/          # CSS + JS
└── tests/                   # pytest test suite
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/documents` | Upload document |
| `GET` | `/api/documents` | List documents |
| `DELETE` | `/api/documents/{id}` | Delete document |
| `POST` | `/api/query` | Ask question (JSON response) |
| `POST` | `/api/query/stream` | Ask question (SSE stream) |
| `GET` | `/api/conversations/{id}` | Get conversation history |
| `GET` | `/api/trace/{id}` | Get query trace |

## Tech Stack

Python 3.11+ · LangGraph · ChromaDB · FastAPI · ARQ + Redis · LangFuse · BGE-M3 · Jinja2
