# rag-app/rag_app/api.py
import html
import re
import uuid
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from rag_core.types import (
    RetrievalQuery, MetadataFilter, Message, QueryResult,
    IngestionResult, DEFAULT_NAMESPACE,
)
from rag_app.graph import build_rag_graph, classify_route, RAGState
from rag_app.deps import (
    get_retriever, get_indexer, get_embedder, get_model,
    get_conversation_store, get_config, get_checkpointer, get_sql_engine,
)

router = APIRouter()

# Trace mapping: trace_id -> {"url": str, "detail": dict}
_trace_store: dict[str, dict] = {}

# Job status store: job_id -> {"status": str, "filename": str, "chunk_count": int}
_job_store: dict[str, dict] = {}


def _is_htmx(request: Request | None) -> bool:
    if request is None:
        return False
    return request.headers.get("HX-Request") == "true"


async def _parse_body(req: Request) -> dict:
    """Parse request body as JSON or form-data, returning a dict."""
    content_type = req.headers.get("content-type", "")
    if "application/json" in content_type:
        return await req.json()
    form = await req.form()
    return dict(form)


@router.get("/health")
async def health():
    return {"status": "ok"}


MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    force: bool = Query(False),
    namespace: str = Query(DEFAULT_NAMESPACE),
    request: Request = None,
):
    safe_name = re.sub(r'[^\w.-]', '_', file.filename or "upload")
    tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}_{safe_name}"

    total = 0
    with open(tmp_path, "wb") as f:
        while chunk := await file.read(65536):
            total += len(chunk)
            if total > MAX_UPLOAD_SIZE:
                f.close()
                tmp_path.unlink()
                raise HTTPException(status_code=413, detail="File too large (max 50 MB)")
            f.write(chunk)

    # Pre-compute source_id so we can poll status after async ingestion
    from rag_core.hashing import hash_file
    content_hash = hash_file(tmp_path)
    source_id = f"{namespace}:{content_hash}"

    # Try ARQ async ingestion first, fall back to sync
    try:
        from arq import create_pool
        from arq.connections import RedisSettings
        from rag_app.tasks import ingest_document

        redis = await create_pool(
            RedisSettings.from_dsn(get_config().redis.url)
        )
        job = await redis.enqueue_job(
            "ingest_document", str(tmp_path), namespace, force,
            _job_id=source_id,
        )
        await redis.close()

        if job is None:
            # Duplicate job — already queued or completed
            tmp_path.unlink(missing_ok=True)
            if _is_htmx(request):
                return HTMLResponse(f"""<div class="upload-result upload-duplicate">
<span class="upload-icon">&#x26A0;</span>
<div><strong>{html.escape(safe_name)}</strong><br><small>Already queued for processing</small></div>
</div>""")
            return {"job_id": source_id, "status": "duplicate"}

        _job_store[source_id] = {"status": "pending", "filename": safe_name}
        if _is_htmx(request):
            return HTMLResponse(f"""<div class="upload-result upload-success">
<span class="upload-icon">&#x23F3;</span>
<div><strong>{html.escape(safe_name)}</strong><br><small>Queued for processing...</small></div>
</div>""")
        return {"job_id": source_id, "status": "pending"}
    except Exception:
        # Fallback: synchronous ingestion
        try:
            from rag_core.ingestion.pipeline import ingest
            result = await ingest(
                tmp_path, get_indexer(), get_embedder(),
                namespace=namespace, force=force,
            )
            _job_store[source_id] = {
                "status": result.status, "chunk_count": result.chunk_count,
                "filename": safe_name,
            }
            if _is_htmx(request):
                ok = result.status in ("success", "updated")
                return HTMLResponse(f"""<div class="upload-result upload-{'success' if ok else 'duplicate'}">
<span class="upload-icon">{'&#x2713;' if ok else '&#x26A0;'}</span>
<div><strong>{html.escape(safe_name)}</strong><br><small>{'Indexed ' + str(result.chunk_count) + ' chunks' if ok else 'File already indexed (skipped)'}</small></div>
</div>""")
            return {"job_id": source_id, "status": result.status, "chunk_count": result.chunk_count}
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


@router.get("/api/documents")
async def list_documents(namespace: str = Query(DEFAULT_NAMESPACE)):
    return await get_indexer().list_sources(namespace)


@router.delete("/api/documents/{source_id}")
async def delete_document(source_id: str, namespace: str = Query(DEFAULT_NAMESPACE)):
    await get_indexer().delete(source_id, namespace)
    return {"status": "deleted"}


@router.get("/api/documents/{source_id}/status")
async def document_status(source_id: str, namespace: str = Query(DEFAULT_NAMESPACE)):
    # Check in-memory job store first
    job = _job_store.get(source_id)
    if job and job.get("status") == "pending":
        # Check if the document has appeared in the index (worker may have finished)
        exists = await get_indexer().source_exists(source_id, namespace)
        if exists:
            job["status"] = "success"
            return {"source_id": source_id, "status": "success", "chunk_count": job.get("chunk_count", 0)}
        return {"source_id": source_id, "status": "pending"}
    if job:
        return {"source_id": source_id, "status": job.get("status"), "chunk_count": job.get("chunk_count", 0)}
    # Check index directly
    exists = await get_indexer().source_exists(source_id, namespace)
    return {"source_id": source_id, "exists": exists, "status": "complete" if exists else "unknown"}


@router.post("/api/query")
async def query(req: Request):
    body = await _parse_body(req)
    if "question" not in body:
        raise HTTPException(status_code=422, detail="Field 'question' is required")
    question = body["question"]
    filters = None
    if "filters" in body and body["filters"]:
        f = body["filters"]
        filters = MetadataFilter(**f)

    retrieval_query = RetrievalQuery(
        text=question,
        namespace=body.get("namespace", DEFAULT_NAMESPACE),
        filters=filters,
    )

    conversation_id = body.get("conversation_id", "default")
    store = get_conversation_store()
    trace_id = uuid.uuid4().hex

    # Classify route first, then dispatch to the right graph
    route_info = classify_route(question)
    selected_route = route_info["route"]

    # Respect structured_query.enabled config + check tables exist
    if selected_route == "structured_query":
        if not get_config().structured_query.enabled:
            selected_route = "production_rag"
            route_info["route_fallback_reason"] = "structured_query disabled in config — using RAG"
        else:
            engine = get_sql_engine()
            try:
                tables = engine.get_tables()
            except Exception:
                tables = []
            if not tables and not get_config().structured_query.ddl:
                selected_route = "production_rag"
                route_info["route_fallback_reason"] = "no tables configured for structured_query — using RAG"

    if selected_route == "structured_query":
        from rag_app.graphs.structured_query import (
            build_structured_query_graph, StructuredQueryState,
        )
        engine = get_sql_engine()
        graph = build_structured_query_graph(
            get_model(), engine,
            table_schema=get_config().structured_query.ddl,
        )
        state: StructuredQueryState = {"question": question, "route": "structured_query"}
        result = await graph.ainvoke(state)
        answer = result.get("answer", "Query returned no results.")
        chunks = []
        structured_result = result.get("result")
    else:
        # Production RAG (or fallback from unimplemented route)
        graph = build_rag_graph(get_retriever(), get_model())
        history = store.get(conversation_id, n=6)
        rag_state: RAGState = {
            "question": question, "retrieval_query": retrieval_query,
            "messages": history,
            "route": selected_route,
            "intended_route": route_info.get("intended_route", selected_route),
            "route_fallback_reason": route_info.get("route_fallback_reason", ""),
        }

        callbacks, handler = _get_callbacks(trace_id)
        result = await graph.ainvoke(rag_state, {"callbacks": callbacks})
        _store_trace(trace_id, handler)
        answer = result.get("answer", "I cannot confidently answer this question based on the available documents.")
        chunks = result.get("chunks", [])
        structured_result = None

    store.add(conversation_id, Message(role="user", content=question))
    store.add(conversation_id, Message(role="assistant", content=answer))

    if _is_htmx(req):
        sources_html = ""
        if chunks:
            sources_html = "<div class=\"sources\">" + "".join(
                f"<span class=\"source-chip\" title=\"{html.escape(c.content[:100])}\">&#x1F4C4; {html.escape(c.source_id[:12])}</span>"
                for c in chunks[:5]
            ) + "</div>"
        return HTMLResponse(f"""<div class="message user"><p>{html.escape(question)}</p></div>
<div class="message assistant">{html.escape(answer)}{sources_html}</div>""")

    route = result.get("route", selected_route) if selected_route != "structured_query" else selected_route
    intended_route = result.get("intended_route", route)
    fallback_reason = result.get("route_fallback_reason", "")
    resp = {
        "answer": answer,
        "route": route,
        "intended_route": intended_route,
        "route_fallback_reason": fallback_reason,
        "sources": [{"content": c.content, "source_id": c.source_id, "score": c.score} for c in chunks],
        "trace_id": trace_id,
        "tokens": {"prompt": 0, "completion": 0},
    }
    if structured_result is not None:
        resp["structured_result"] = {
            "columns": structured_result.columns,
            "rows": structured_result.rows[:50],
            "row_count": structured_result.row_count,
        }
    return resp


@router.post("/api/query/stream")
async def query_stream(req: Request):
    body = await _parse_body(req)
    question = body.get("question", "")
    if not question.strip():
        raise HTTPException(status_code=422, detail="Field 'question' is required")
    conversation_id = body.get("conversation_id", "default")

    async def event_generator():
        yield {"event": "connected", "data": ""}

        # Classify route first
        route_info = classify_route(question)
        selected_route = route_info["route"]
        cfg = get_config()

        # Respect structured_query.enabled config + check tables exist
        if selected_route == "structured_query":
            if not cfg.structured_query.enabled:
                selected_route = "production_rag"
                route_info["route_fallback_reason"] = "structured_query disabled in config — using RAG"
            else:
                engine = get_sql_engine()
                try:
                    tables = engine.get_tables()
                except Exception:
                    tables = []
                if not tables and not cfg.structured_query.ddl:
                    selected_route = "production_rag"
                    route_info["route_fallback_reason"] = "no tables configured for structured_query — using RAG"

        if selected_route == "structured_query":
            # Structured query path
            from rag_app.graphs.structured_query import build_structured_query_graph
            engine = get_sql_engine()
            graph = build_structured_query_graph(
                get_model(), engine, table_schema=cfg.structured_query.ddl
            )
            yield {"event": "route", "data": "structured_query"}

            try:
                async for chunk in graph.astream({"question": question, "route": "structured_query"}):
                    for node_name, node_data in chunk.items():
                        if node_name == "generate_sql":
                            yield {"event": "step", "data": "Generating SQL..."}
                        elif node_name == "execute_readonly":
                            yield {"event": "step", "data": "Executing query..."}
                        elif node_name == "explain" and "answer" in node_data:
                            answer = node_data.get("answer", "")
                            for char in answer:
                                yield {"event": "token", "data": char}
            except Exception as e:
                import traceback, sys
                print(f"Stream SQL error: {e}", file=sys.stderr)
                traceback.print_exc()
                yield {"event": "error", "data": f"SQL query failed: {e}"}
                return

            yield {"event": "done", "data": ""}
            store = get_conversation_store()
            store.add(conversation_id, Message(role="user", content=question))
            store.add(conversation_id, Message(role="assistant", content=answer))
            return

        # Production RAG path
        yield {"event": "route", "data": selected_route}
        if route_info.get("route_fallback_reason"):
            yield {"event": "route_fallback", "data": route_info["route_fallback_reason"]}

        graph = build_rag_graph(get_retriever(), get_model())
        store = get_conversation_store()
        retrieval_query = RetrievalQuery(
            text=question,
            namespace=body.get("namespace", DEFAULT_NAMESPACE),
        )
        history = store.get(conversation_id, n=6)
        state: RAGState = {
            "question": question, "retrieval_query": retrieval_query,
            "messages": history,
        }
        answer = ""
        chunks: list = []

        node_labels = {
            "classify_route": "",
            "rewrite": "Rewriting query...",
            "retrieve": "Searching documents...",
            "generate": "Generating answer...",
            "check": "Verifying accuracy...",
        }
        try:
            stream_callbacks, _ = _get_callbacks()
            async for chunk in graph.astream(state, {"callbacks": stream_callbacks}):
                for node_name, node_data in chunk.items():
                    if node_name == "classify_route":
                        route = node_data.get("intended_route", node_data.get("route", "production_rag"))
                        fallback = node_data.get("route_fallback_reason", "")
                        yield {"event": "route", "data": route}
                        if fallback:
                            yield {"event": "route_fallback", "data": fallback}
                        continue
                    label = node_labels.get(node_name, node_name)
                    yield {"event": "step", "data": label}
                    if node_name == "retrieve" and "chunks" in node_data:
                        chunks = node_data["chunks"]
                        for c in chunks[:5]:
                            yield {"event": "source", "data": c.content[:150]}
                    if node_name == "generate" and "answer" in node_data:
                        answer = node_data.get("answer", "")
        except Exception as e:
            import traceback, sys
            print(f"Stream error: {e}", file=sys.stderr)
            traceback.print_exc()
            yield {"event": "error", "data": f"Query failed: {e}"}
            return

        # Stream answer characters for typewriter effect
        for char in answer:
            yield {"event": "token", "data": char}
        yield {"event": "done", "data": ""}

        # Save to conversation
        store.add(conversation_id, Message(role="user", content=question))
        store.add(conversation_id, Message(role="assistant", content=answer))

    return EventSourceResponse(event_generator())


@router.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    store = get_conversation_store()
    msgs = store.get(conversation_id)
    return [{"role": m.role, "content": m.content, "name": m.name} for m in msgs]


@router.get("/api/trace/{trace_id}")
async def get_trace(trace_id: str):
    stored = _trace_store.get(trace_id)
    if stored:
        return {
            "trace_id": trace_id,
            "langfuse_trace_url": stored.get("url"),
            "langfuse_trace_id": stored.get("langfuse_trace_id"),
        }
    return {
        "trace_id": trace_id,
        "detail": "Trace not found. It may have expired or tracing is disabled.",
    }


@router.get("/api/settings")
async def get_settings():
    cfg = get_config()
    return {
        "llm_provider": cfg.llm.provider,
        "llm_model": cfg.llm.model,
        "llm_api_key": cfg.llm.api_key[:8] + "..." if len(cfg.llm.api_key) > 8 else "",
        "llm_base_url": cfg.llm.base_url,
        "embedding_provider": cfg.embedding.provider,
        "embedding_model": cfg.embedding.model,
        "embedding_api_key": cfg.embedding.api_key[:8] + "..." if len(cfg.embedding.api_key) > 8 else "",
        "reranker_enabled": cfg.reranker.enabled,
    }


@router.post("/api/settings")
async def save_settings(req: Request):
    import yaml
    body = await _parse_body(req)
    cfg = get_config()
    config_path = Path("config.yaml")

    if "llm_provider" in body:
        cfg.llm.provider = body["llm_provider"]
    if "llm_model" in body:
        cfg.llm.model = body["llm_model"]
    if "llm_api_key" in body and body["llm_api_key"] and not body["llm_api_key"].endswith("..."):
        cfg.llm.api_key = body["llm_api_key"]
    if "llm_base_url" in body:
        cfg.llm.base_url = body["llm_base_url"]
    if "embedding_provider" in body:
        cfg.embedding.provider = body["embedding_provider"]
    if "embedding_model" in body:
        cfg.embedding.model = body["embedding_model"]
    if "embedding_api_key" in body and body["embedding_api_key"] and not body["embedding_api_key"].endswith("..."):
        cfg.embedding.api_key = body["embedding_api_key"]
    if "reranker_enabled" in body:
        cfg.reranker.enabled = body["reranker_enabled"] in (True, "true", "on")

    raw = {
        "namespace": cfg.namespace,
        "llm": {
            "provider": cfg.llm.provider,
            "model": cfg.llm.model,
            "api_key": "${LLM_API_KEY}",
            "base_url": cfg.llm.base_url,
        },
        "embedding": {
            "provider": cfg.embedding.provider,
            "model": cfg.embedding.model,
            "api_key": "${EMBEDDING_API_KEY}",
        },
        "reranker": {"enabled": cfg.reranker.enabled},
        "chromadb": {
            "persist_dir": cfg.chromadb.persist_dir,
            "collection_name": cfg.chromadb.collection_name,
        },
        "vector_store": {
            "backend": cfg.vector_store.backend,
            "chromadb": {
                "persist_dir": cfg.vector_store.chromadb.persist_dir,
                "collection_name": cfg.vector_store.chromadb.collection_name,
            },
            "qdrant": {
                "url": cfg.vector_store.qdrant.url,
                "collection_name": cfg.vector_store.qdrant.collection_name,
            },
        },
        "structured_query": {
            "enabled": cfg.structured_query.enabled,
            "db_path": cfg.structured_query.db_path,
            "ddl": cfg.structured_query.ddl,
        },
        "redis": {"url": cfg.redis.url},
        "server": {"host": cfg.server.host, "port": cfg.server.port},
        "langfuse": {
            "enabled": cfg.langfuse.enabled,
            "public_key": cfg.langfuse.public_key,
            "secret_key": cfg.langfuse.secret_key,
            "host": cfg.langfuse.host,
        },
    }
    config_path.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")
    return HTMLResponse("<span class=\"save-ok\">Saved. Restart to apply LLM changes.</span>")


def _get_callbacks(trace_id: str | None = None):
    """Return LangFuse callback handler if enabled in config, else empty list.
    Returns (callbacks, handler) tuple for trace URL extraction.
    """
    handler = None
    try:
        cfg = get_config()
        if cfg.langfuse.enabled:
            from langfuse.callback import CallbackHandler
            handler = CallbackHandler(trace_id=trace_id)
            return [handler], handler
    except Exception:
        pass
    return [], None


def _store_trace(trace_id: str, handler):
    """Extract and store trace URL/ID from LangFuse handler."""
    cfg = get_config()
    try:
        langfuse_trace_id = getattr(handler, 'trace_id', None) or trace_id
        trace_url = f"{cfg.langfuse.host.rstrip('/')}/trace/{langfuse_trace_id}"
    except Exception:
        trace_url = None
    _trace_store[trace_id] = {
        "url": trace_url,
        "langfuse_trace_id": langfuse_trace_id if handler else None,
    }
