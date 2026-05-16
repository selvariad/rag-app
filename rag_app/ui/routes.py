# rag-app/rag_app/ui/routes.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from rag_app.deps import get_config

ui_router = APIRouter()

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_jinja = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))


def _config_context():
    cfg = get_config()
    return {
        "llm_provider": cfg.llm.provider,
        "llm_model": cfg.llm.model,
        "llm_base_url": cfg.llm.base_url,
        "embedding_provider": cfg.embedding.provider,
        "embedding_model": cfg.embedding.model,
        "reranker_enabled": cfg.reranker.enabled,
        "namespace": cfg.namespace,
    }


@ui_router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    ctx = _config_context()
    template = _jinja.get_template("index.html")
    return HTMLResponse(template.render(**ctx))


@ui_router.get("/documents", response_class=HTMLResponse)
async def documents_page(request: Request):
    ctx = _config_context()
    template = _jinja.get_template("documents.html")
    return HTMLResponse(template.render(**ctx))


@ui_router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    ctx = _config_context()
    template = _jinja.get_template("settings.html")
    return HTMLResponse(template.render(**ctx))
