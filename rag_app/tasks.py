# rag-app/rag_app/tasks.py
from pathlib import Path
from rag_core.ingestion.pipeline import ingest
from rag_core.types import IngestionResult
from rag_app.deps import get_indexer, get_embedder


async def ingest_document(ctx, source_path: str, namespace: str = "default", force: bool = False) -> dict:
    path = Path(source_path)
    result: IngestionResult = await ingest(
        path,
        get_indexer(),
        get_embedder(),
        namespace=namespace,
        force=force,
    )
    return {
        "source_id": result.source_id,
        "chunk_count": result.chunk_count,
        "status": result.status,
    }
