# rag-app/rag_app/ui/routes.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pathlib import Path

ui_router = APIRouter()

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


@ui_router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _TEMPLATE_PATH.read_text(encoding="utf-8")
