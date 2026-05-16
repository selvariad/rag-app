# rag-app/rag_app/api.py
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
from rag_app.graph import build_rag_graph, RAGState
from rag_app.deps import (
    get_retriever, get_indexer, get_embedder, get_model,
    get_conversation_store, get_config,
)

router = APIRouter()


def _is_htmx(request: Request | None) -> bool:
    if request is None:
        return False
    return request.headers.get("HX-Request") == "true"


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    force: bool = Query(False),
    namespace: str = Query(DEFAULT_NAMESPACE),
    request: Request = None,
):
    content = await file.read()
    tmp_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}_{file.filename}"
    tmp_path.write_bytes(content)
    try:
        from rag_core.ingestion.pipeline import ingest
        result = await ingest(
            tmp_path,
            get_indexer(),
            get_embedder(),
            namespace=namespace,
            force=force,
        )
        if _is_htmx(request):
            status_class = "success" if result.status in ("success", "updated") else "duplicate"
            icon = "✓" if status_class == "success" else "⚠"
            msg = f"Indexed {result.chunk_count} chunks" if status_class == "success" else "File already indexed (skipped)"
            return HTMLResponse(f"""<div class="upload-result upload-{status_class}">
<span class="upload-icon">{icon}</span>
<div><strong>{file.filename}</strong><br><small>{msg}</small></div>
</div>""")
        return {"job_id": result.source_id, "status": result.status, "chunk_count": result.chunk_count}
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.get("/api/documents")
async def list_documents(namespace: str = Query(DEFAULT_NAMESPACE)):
    return []


@router.delete("/api/documents/{source_id}")
async def delete_document(source_id: str, namespace: str = Query(DEFAULT_NAMESPACE)):
    await get_indexer().delete(source_id, namespace)
    return {"status": "deleted"}


@router.get("/api/documents/{source_id}/status")
async def document_status(source_id: str, namespace: str = Query(DEFAULT_NAMESPACE)):
    exists = await get_indexer().source_exists(source_id, namespace)
    return {"source_id": source_id, "exists": exists}


@router.post("/api/query")
async def query(request: dict, req: Request = None):
    if "question" not in request:
        raise HTTPException(status_code=422, detail="Field 'question' is required")
    question = request["question"]
    filters = None
    if "filters" in request and request["filters"]:
        f = request["filters"]
        filters = MetadataFilter(**f)

    retrieval_query = RetrievalQuery(
        text=question,
        namespace=request.get("namespace", DEFAULT_NAMESPACE),
        filters=filters,
    )

    conversation_id = request.get("conversation_id", "default")
    store = get_conversation_store()

    graph = build_rag_graph(get_retriever(), get_model())
    trace_id = uuid.uuid4().hex

    state: RAGState = {"question": question, "retrieval_query": retrieval_query}
    result = await graph.ainvoke(state)

    answer = result.get("answer", "I cannot confidently answer this question based on the available documents.")
    chunks = result.get("chunks", [])

    store.add(conversation_id, Message(role="user", content=question))
    store.add(conversation_id, Message(role="assistant", content=answer))

    if _is_htmx(req):
        sources_html = ""
        if chunks:
            sources_html = "<div class=\"sources\">" + "".join(
                f"<span class=\"source-chip\" title=\"{c.content[:100]}\">\U0001F4C4 {c.source_id[:12]}</span>"
                for c in chunks[:5]
            ) + "</div>"
        return HTMLResponse(f"""<div class="message user"><p>{question}</p></div>
<div class="message assistant">{answer}{sources_html}</div>""")

    return {
        "answer": answer,
        "sources": [{"content": c.content, "source_id": c.source_id, "score": c.score} for c in chunks],
        "trace_id": trace_id,
        "tokens": {"prompt": 0, "completion": 0},
    }


@router.post("/api/query/stream")
async def query_stream(request: dict):
    async def event_generator():
        question = request["question"]
        model = get_model()
        graph = build_rag_graph(get_retriever(), model)
        state: RAGState = {"question": question}
        result = await graph.ainvoke(state)
        answer = result.get("answer", "")
        chunks = result.get("chunks", [])

        for source in chunks:
            yield {"event": "source", "data": source.content[:200]}
        for char in answer:
            yield {"event": "token", "data": char}
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


@router.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    store = get_conversation_store()
    msgs = store.get(conversation_id)
    return [{"role": m.role, "content": m.content, "name": m.name} for m in msgs]


@router.get("/api/trace/{trace_id}")
async def get_trace(trace_id: str):
    return {"trace_id": trace_id, "detail": "Trace data available in LangFuse dashboard."}


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
async def save_settings(request: dict):
    import yaml
    cfg = get_config()
    config_path = Path("config.yaml")

    if "llm_provider" in request:
        cfg.llm.provider = request["llm_provider"]
    if "llm_model" in request:
        cfg.llm.model = request["llm_model"]
    if "llm_api_key" in request and request["llm_api_key"] and not request["llm_api_key"].endswith("..."):
        cfg.llm.api_key = request["llm_api_key"]
    if "llm_base_url" in request:
        cfg.llm.base_url = request["llm_base_url"]
    if "embedding_provider" in request:
        cfg.embedding.provider = request["embedding_provider"]
    if "embedding_model" in request:
        cfg.embedding.model = request["embedding_model"]
    if "embedding_api_key" in request and request["embedding_api_key"] and not request["embedding_api_key"].endswith("..."):
        cfg.embedding.api_key = request["embedding_api_key"]
    if "reranker_enabled" in request:
        cfg.reranker.enabled = request["reranker_enabled"] in (True, "true", "on")

    raw = {
        "namespace": cfg.namespace,
        "llm": {
            "provider": cfg.llm.provider,
            "model": cfg.llm.model,
            "api_key": cfg.llm.api_key,
            "base_url": cfg.llm.base_url,
        },
        "embedding": {
            "provider": cfg.embedding.provider,
            "model": cfg.embedding.model,
            "api_key": cfg.embedding.api_key,
        },
        "reranker": {"enabled": cfg.reranker.enabled},
        "chromadb": {
            "persist_dir": cfg.chromadb.persist_dir,
            "collection_name": cfg.chromadb.collection_name,
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
