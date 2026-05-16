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
