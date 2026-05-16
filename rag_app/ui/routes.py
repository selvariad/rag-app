# rag-app/rag_app/ui/routes.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from rag_app.deps import get_config

ui_router = APIRouter()

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_jinja = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))


@ui_router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    cfg = get_config()
    template = _jinja.get_template("index.html")
    return HTMLResponse(template.render(model=f"{cfg.llm.provider}/{cfg.llm.model}"))


@ui_router.get("/documents", response_class=HTMLResponse)
async def documents_redirect():
    return RedirectResponse(url="/?view=documents", status_code=302)
