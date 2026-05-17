# RAG Enterprise Document Assistant

Enterprise knowledge Q&A platform with multi-route architecture — not everything goes through RAG. Built on [rag-core](https://github.com/selvariad/rag-core).

## Features

### Knowledge Routes

| Route | Trigger | Status |
|-------|---------|--------|
| `production_rag` | Document questions, policy lookup | ✅ Production |
| `structured_query` | SQL/table/metric questions | ✅ Production — SELECT-only, table allowlists |
| `deep_research` | Research, comparison, analysis | 🔜 Fallback to RAG |
| `agentic_retrieval` | Code, debug, multi-hop | 🔜 Fallback to RAG |
| `long_context` | Single file, pasted text | 🔜 Not started |

### RAG Pipeline

- **Hybrid search** — ChromaDB vector + BM25 keyword + optional BGE-Reranker
- **Self-correcting graph** — LangGraph 5-node state machine (classify → rewrite → retrieve → generate → check) with conditional retry loop
- **Two failure modes** — distinguishes `model_strayed` (re-generate with stricter prompt) vs `chunks_irrelevant` (rewrite query)
- **Inline citations** — model instructed to cite `[Source N]` in answers
- **Metadata extraction** — title, date, author heuristics during ingestion

### Structured Query

- **SQLiteQueryEngine** — read-only SQL with table allowlist, column allowlist, auto LIMIT
- **LLM generates SQL** from natural language + schema
- **Safety layers**: SELECT-only, table allowlist, default LIMIT 100, validation errors returned as-is (not silently falling back)
- **Graceful fallback**: no tables configured → RAG; execution failure → RAG; validation failure → error to user

### Platform

- **Streaming SSE** — token-by-token response with reasoning chain display
- **Route-aware UI** — shows which knowledge path was selected (RAG / SQL / Agent / Research)
- **Async ingestion** — ARQ background worker + sync fallback, 50MB limit, filename sanitization
- **Conversation memory** — multi-turn chat with history injection into graph state
- **Observability** — LangFuse callback wiring, trace API
- **Dark/Light theme** — Gemini-inspired responsive UI
- **Standalone config editor** — `config.html` multi-profile API manager with Apply button

## Quick Start

```bash
# Install rag-core
cd ../rag-core && pip install -e ".[all]"

# Install rag-app
cd ../rag-app && pip install -e ".[dev]"

# Configure (edit config.yaml or use config.html)
# config.yaml is gitignored — copy from config.example.yaml first
cp config.example.yaml config.yaml

# Set your API key
# Option A: environment variable
export LLM_API_KEY=sk-your-key

# Option B: edit config.yaml directly (gitignored)

# Option C: use the visual config editor
open config.html

# Start
python main.py
# → http://127.0.0.1:8000
```

## Configuration

Copy `config.example.yaml` to `config.yaml` and edit. Key sections:

```yaml
llm:
  provider: deepseek        # openai | anthropic | deepseek | zhipu
  model: deepseek-chat
  api_key: ${LLM_API_KEY}   # env var or literal key

structured_query:
  enabled: true
  db_path: "./data.db"
  ddl: "CREATE TABLE users (...)"   # optional schema
  table_allowlist: []               # [] = deny all, ["users"] = whitelist
  default_limit: 100
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/documents` | Upload document (ARQ async + sync fallback) |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{id}` | Delete document and chunks |
| `GET` | `/api/documents/{id}/status` | Ingestion job status |
| `POST` | `/api/query` | Query — returns `{answer, route, route_reason, sources, trace_id}` |
| `POST` | `/api/query/stream` | SSE stream with route/step/token/done events |
| `GET` | `/api/conversations/{id}` | Conversation history |
| `GET` | `/api/trace/{id}` | Trace with LangFuse deep link |

## Project Structure

```
rag-app/
├── main.py                     # Entry point
├── config.example.yaml         # Safe config template (no secrets)
├── config.yaml                 # Local config (gitignored)
├── config.html                 # Standalone multi-profile API manager
├── rag_app/
│   ├── config.py               # Config dataclasses + YAML/env loader
│   ├── deps.py                 # Dependency injection
│   ├── graph.py                # RAG state machine + route classifier
│   ├── graphs/
│   │   └── structured_query.py # SQL generation + execution + explain
│   ├── api.py                  # FastAPI routes with route dispatch
│   ├── app.py                  # App factory + lifespan (checkpointer)
│   ├── tasks.py                # ARQ background ingestion
│   ├── worker.py               # ARQ worker process
│   └── ui/                     # Jinja2 templates + CSS + JS
└── tests/                      # 36 tests
```

## Tech Stack

Python 3.11+ · LangGraph · ChromaDB · SQLite · FastAPI · ARQ + Redis · LangFuse · BGE-M3 · Jinja2
