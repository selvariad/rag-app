# rag-app/rag_app/api.py
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
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


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    force: bool = Query(False),
    namespace: str = Query(DEFAULT_NAMESPACE),
):
    content = await file.read()
    tmp_path = Path(f"/tmp/{uuid.uuid4().hex}_{file.filename}")
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
async def query(request: dict):
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
