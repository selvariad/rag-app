# rag-app/tests/test_api.py
import tempfile
import pytest
from httpx import AsyncClient, ASGITransport
from rag_app.app import create_app
from rag_app.config import AppConfig, ChromaDBConfig, VectorStoreConfig
from rag_app import deps
from rag_core.types import Message


class FakeModel:
    """Fake chat model that returns a canned answer, used in tests to avoid real LLM API calls."""

    def __init__(self, answer: str = "PTO is 15 days per year."):
        self._answer = answer
        self.last_messages = None

    async def invoke(self, messages: list[Message], tools=None) -> Message:
        self.last_messages = messages
        return Message(role="assistant", content=self._answer)

    async def stream(self, messages: list[Message]):
        for char in self._answer:
            yield char


@pytest.fixture
def app():
    tmpdir = tempfile.mkdtemp()
    chroma_cfg = ChromaDBConfig(persist_dir=tmpdir, collection_name="test_documents")
    cfg = AppConfig(
        namespace="default",
        chromadb=chroma_cfg,
        vector_store=VectorStoreConfig(backend="chromadb", chromadb=chroma_cfg),
    )
    try:
        app_instance = create_app(cfg)
        # Replace the real LLM with a fake to avoid requiring API keys
        deps.set_model(FakeModel())
        return app_instance
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"Required dependency not available: {e}")


@pytest.mark.asyncio
async def test_health_check(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_list_documents_empty(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/documents")
        assert resp.status_code == 200
        assert resp.json() == []


@pytest.mark.asyncio
async def test_query_endpoint_requires_body(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/query", json={})
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_query_endpoint(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/query", json={
            "question": "What is the PTO policy?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "trace_id" in data


@pytest.mark.asyncio
async def test_conversation_persistence(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/query", json={
            "question": "What is PTO?",
            "conversation_id": "test_conv",
        })
        resp = await client.get("/api/conversations/test_conv")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
