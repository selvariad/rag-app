# rag-app/rag_app/app.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from rag_app.ui.routes import ui_router


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Enterprise Assistant", version="0.1.0")

    ui_dir = Path(__file__).parent / "ui"
    app.mount("/static", StaticFiles(directory=str(ui_dir / "static")), name="static")
    app.include_router(ui_router)

    return app
