# rag-app/rag_app/app.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from rag_app.api import router
from rag_app.ui.routes import ui_router
from rag_app.config import AppConfig
from rag_app import deps


def create_app(config: AppConfig) -> FastAPI:
    deps.init(config)
    app = FastAPI(title="RAG Enterprise Document Assistant")
    ui_dir = Path(__file__).parent / "ui"
    app.mount("/static", StaticFiles(directory=str(ui_dir / "static")), name="static")
    app.include_router(ui_router)
    app.include_router(router)
    return app
