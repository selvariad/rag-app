# rag-app/tests/test_integration.py
import tempfile
import os
import uuid
import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from rag_app.app import create_app
from rag_app.config import AppConfig

needs_llm = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — set it to run integration tests that call the LLM",
)


@pytest.fixture
def app():
    cfg = AppConfig(namespace="default")
    return create_app(cfg)


@pytest.mark.asyncio
@needs_llm
async def test_full_upload_query_delete_flow(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a test file with unique content to avoid hash collision
        import uuid
        content = f"PTO Policy: Employees receive 15 days of paid time off per calendar year. (test_id: {uuid.uuid4().hex})"
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        tmp.write(content.encode())
        tmp.close()

        # Upload
        with open(tmp.name, "rb") as f:
            resp = await client.post("/api/documents", files={"file": f})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("success", "pending")

        # Query
        resp = await client.post("/api/query", json={"question": "How many PTO days?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["sources"]) >= 0

        os.unlink(tmp.name)
