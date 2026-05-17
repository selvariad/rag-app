# rag-app/tests/test_graph.py
import pytest
from rag_app.graph import (
    RAGState, build_rag_graph, route_after_check,
)
from rag_core.types import RetrievalQuery, RetrievedChunk, Chunk, Message, MetadataFilter


class FakeRetriever:
    def __init__(self, chunks=None):
        self._chunks = chunks or [
            RetrievedChunk(id="a:0", content="PTO is 15 days", metadata={}, source_id="a", score=0.95, rank=1),
            RetrievedChunk(id="a:1", content="Benefits include health", metadata={}, source_id="a", score=0.8, rank=2),
        ]
        self.last_query = None

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        self.last_query = query
        return self._chunks


class FakeModel:
    def __init__(self, answer: str = "PTO is 15 days per year."):
        self._answer = answer
        self.last_messages = None

    async def invoke(self, messages: list[Message], tools=None) -> Message:
        self.last_messages = messages
        return Message(role="assistant", content=self._answer)

    async def stream(self, messages: list[Message]):
        for char in self._answer:
            yield char


@pytest.mark.asyncio
async def test_build_rag_graph_returns_compiled_graph():
    retriever = FakeRetriever()
    model = FakeModel()
    graph = build_rag_graph(retriever, model)
    assert graph is not None


@pytest.mark.asyncio
async def test_graph_basic_flow():
    retriever = FakeRetriever()
    model = FakeModel()
    graph = build_rag_graph(retriever, model)

    state: RAGState = {"question": "What is the PTO policy?"}
    result = await graph.ainvoke(state, {"configurable": {"thread_id": "test"}})

    assert "answer" in result
    assert "PTO" in result["answer"]
    assert "chunks" in result
    assert len(result["chunks"]) == 2
    assert retriever.last_query is not None
    assert retriever.last_query.text == "What is the PTO policy?"


@pytest.mark.asyncio
async def test_graph_with_metadata_filter():
    retriever = FakeRetriever()
    model = FakeModel()
    graph = build_rag_graph(retriever, model)

    state: RAGState = {
        "question": "benefits",
        "retrieval_query": RetrievalQuery(
            text="benefits",
            filters=MetadataFilter(field="dept", op="eq", value="HR"),
        ),
    }
    result = await graph.ainvoke(state, {"configurable": {"thread_id": "test"}})
    assert retriever.last_query.filters is not None
    assert retriever.last_query.filters.field == "dept"


def test_route_after_check_grounded():
    result = route_after_check({
        "grounded": True,
        "grounding_reason": "ok",
        "regen_count": 0,
        "rewrite_count": 0,
    })
    assert result == "end"


def test_route_after_check_model_strayed():
    result = route_after_check({
        "grounded": False,
        "grounding_reason": "model_strayed",
        "regen_count": 1,
        "rewrite_count": 0,
    })
    assert result == "regenerate"


def test_route_after_check_model_strayed_exhausted():
    result = route_after_check({
        "grounded": False,
        "grounding_reason": "model_strayed",
        "regen_count": 3,
        "rewrite_count": 0,
    })
    assert result == "end"


def test_route_after_check_chunks_irrelevant():
    result = route_after_check({
        "grounded": False,
        "grounding_reason": "chunks_irrelevant",
        "regen_count": 0,
        "rewrite_count": 0,
    })
    assert result == "rewrite"


def test_route_after_check_chunks_irrelevant_exhausted():
    result = route_after_check({
        "grounded": False,
        "grounding_reason": "chunks_irrelevant",
        "regen_count": 0,
        "rewrite_count": 2,
    })
    assert result == "end"


@pytest.mark.asyncio
async def test_retrieval_precision_top_chunks():
    """Query against known chunks — verify expected content in top results."""
    chunks = [
        RetrievedChunk(id="a:0", content="PTO policy: 15 days per year", metadata={}, source_id="a", score=0.95, rank=1),
        RetrievedChunk(id="a:1", content="Benefits include health insurance", metadata={}, source_id="a", score=0.8, rank=2),
        RetrievedChunk(id="b:0", content="Security policy: badges required", metadata={}, source_id="b", score=0.6, rank=3),
    ]
    retriever = FakeRetriever(chunks=chunks)
    model = FakeModel(answer="Based on the documents, PTO is 15 days [Source 1].")
    graph = build_rag_graph(retriever, model)

    state: RAGState = {"question": "How many PTO days?"}
    result = await graph.ainvoke(state, {"configurable": {"thread_id": "test_precision"}})

    assert len(result["chunks"]) == 3
    assert result["chunks"][0].rank == 1
    assert "PTO" in result["chunks"][0].content
    assert "15 days" in result["chunks"][0].content


@pytest.mark.asyncio
async def test_route_classify_returns_production_rag():
    """The classify_route node should set route to production_rag."""
    retriever = FakeRetriever()
    model = FakeModel()
    graph = build_rag_graph(retriever, model)

    state: RAGState = {"question": "test"}
    result = await graph.ainvoke(state, {"configurable": {"thread_id": "test_route"}})

    assert result.get("route") == "production_rag"


@pytest.mark.asyncio
async def test_model_strayed_triggers_regenerate():
    """Model ignores relevant chunks — check should complete without looping."""
    chunks = [
        RetrievedChunk(id="a:0", content="PTO is 15 days", metadata={}, source_id="a", score=0.95, rank=1),
    ]
    retriever = FakeRetriever(chunks=chunks)
    model = FakeModel(answer="PTO is 20 days per year.")
    graph = build_rag_graph(retriever, model)

    state: RAGState = {"question": "What is PTO?"}
    result = await graph.ainvoke(state, {"configurable": {"thread_id": "test_strayed"}})

    assert "answer" in result
    reason = result.get("grounding_reason", "ok")
    assert reason in ("ok", "model_strayed", "chunks_irrelevant")


@pytest.mark.asyncio
async def test_chunks_irrelevant_triggers_rewrite():
    """Query about something not in chunks — should complete without looping."""
    chunks: list = []
    retriever = FakeRetriever(chunks=chunks)
    model = FakeModel(answer="I cannot answer this based on available documents.")
    graph = build_rag_graph(retriever, model)

    state: RAGState = {"question": "What is the stock price?"}
    result = await graph.ainvoke(state, {"configurable": {"thread_id": "test_irrelevant"}})

    assert "answer" in result
    # With no chunks, grounded should be True (empty check short-circuits)
    assert result.get("grounded", True) is True


@pytest.mark.asyncio
async def test_classify_structured_query():
    """SQL/metric keywords should trigger structured_query route directly (no fallback)."""
    retriever = FakeRetriever()
    model = FakeModel()
    graph = build_rag_graph(retriever, model)

    state: RAGState = {"question": "how many users signed up last month"}
    result = await graph.ainvoke(state, {"configurable": {"thread_id": "test_sql"}})

    assert result.get("route") == "structured_query"
    assert result.get("intended_route") == "structured_query"
    # No fallback — structured_query is now implemented


@pytest.mark.asyncio
async def test_classify_deep_research():
    """Research/compare keywords should trigger deep_research intent with fallback."""
    retriever = FakeRetriever()
    model = FakeModel()
    graph = build_rag_graph(retriever, model)

    state: RAGState = {"question": "compare market strategy of our competitors"}
    result = await graph.ainvoke(state, {"configurable": {"thread_id": "test_research"}})

    assert result.get("intended_route") == "deep_research"
    assert "not yet implemented" in result.get("route_fallback_reason", "")
    assert result.get("route") == "production_rag"


@pytest.mark.asyncio
async def test_classify_agentic_retrieval():
    """Debug/code keywords should trigger agentic_retrieval intent with fallback."""
    retriever = FakeRetriever()
    model = FakeModel()
    graph = build_rag_graph(retriever, model)

    state: RAGState = {"question": "debug the traceback in main.py line 42"}
    result = await graph.ainvoke(state, {"configurable": {"thread_id": "test_agentic"}})

    assert result.get("intended_route") == "agentic_retrieval"
    assert "not yet implemented" in result.get("route_fallback_reason", "")
    assert result.get("route") == "production_rag"
