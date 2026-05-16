# rag-app/rag_app/ui/routes.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pathlib import Path

ui_router = APIRouter()

_TEMPLATE_DIR = Path(__file__).parent / "templates"


@ui_router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return (_TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")


@ui_router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return (_TEMPLATE_DIR / "settings.html").read_text(encoding="utf-8")
