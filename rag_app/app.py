# rag-app/rag_app/app.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from rag_app.api import router
from rag_app.ui.routes import ui_router
from rag_app.config import AppConfig
from rag_app import deps


def create_app(config: AppConfig) -> FastAPI:
    deps.init(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        serde = JsonPlusSerializer(allowed_msgpack_modules=[
            ("rag_core.types", "RetrievalQuery"),
            ("rag_core.types", "RetrievedChunk"),
            ("rag_core.types", "Chunk"),
            ("rag_core.types", "Message"),
            ("rag_core.types", "MetadataFilter"),
        ])
        async with aiosqlite.connect("checkpoints.db") as conn:
            checkpointer = AsyncSqliteSaver(conn, serde=serde)
            deps.set_checkpointer(checkpointer)
            yield

    app = FastAPI(title="RAG Enterprise Document Assistant", lifespan=lifespan)
    ui_dir = Path(__file__).parent / "ui"
    app.mount("/static", StaticFiles(directory=str(ui_dir / "static")), name="static")
    app.include_router(ui_router)
    app.include_router(router)
    return app
