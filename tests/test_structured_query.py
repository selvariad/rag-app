# rag-app/tests/test_structured_query.py
import pytest
from rag_core.capabilities.structured_query import (
    StructuredQueryEngine, SQLiteQueryEngine,
)
from rag_core.types import StructuredQuery, StructuredResult, Message
from rag_app.graph import classify_route


class FakeSQLModel:
    """Fake model that returns SQL or explanations for structured query tests."""
    def __init__(self, sql: str = "SELECT COUNT(*) FROM users", answer: str = "There are 42 users."):
        self._sql = sql
        self._answer = answer
        self.last_messages = None

    async def invoke(self, messages: list[Message], tools=None) -> Message:
        self.last_messages = messages
        last = messages[-1].content if messages else ""
        if "SQL expert" in last or "Generate" in last:
            return Message(role="assistant", content=self._sql)
        return Message(role="assistant", content=self._answer)

    async def stream(self, messages: list[Message]):
        for char in self._answer:
            yield char


# ── Engine tests ──

def test_sqlite_engine_satisfies_protocol():
    engine = SQLiteQueryEngine()
    assert isinstance(engine, StructuredQueryEngine)


@pytest.mark.asyncio
async def test_select_allowed():
    engine = SQLiteQueryEngine()
    engine.setup_schema("CREATE TABLE users (id INTEGER, name TEXT)")
    engine.setup_schema("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")

    result = await engine.execute_readonly(StructuredQuery(sql="SELECT COUNT(*) FROM users"))
    assert result.row_count == 1
    assert result.rows[0][0] == 2


@pytest.mark.asyncio
async def test_insert_rejected():
    engine = SQLiteQueryEngine()
    engine.setup_schema("CREATE TABLE users (id INTEGER, name TEXT)")

    with pytest.raises(ValueError, match="Only SELECT"):
        await engine.execute_readonly(StructuredQuery(sql="INSERT INTO users VALUES (1, 'test')"))


@pytest.mark.asyncio
async def test_drop_rejected():
    engine = SQLiteQueryEngine()

    with pytest.raises(ValueError, match="Only SELECT"):
        await engine.execute_readonly(StructuredQuery(sql="DROP TABLE users"))


# ── Classifier tests ──

def test_classify_structured_keyword_returns_sql_route():
    result = classify_route("how many users signed up last month")
    assert result["route"] == "structured_query"
    assert result["intended_route"] == "structured_query"


def test_classify_count_returns_sql_route():
    result = classify_route("count the total number of customers")
    assert result["route"] == "structured_query"


def test_classify_metric_returns_sql_route():
    result = classify_route("what is the average revenue per user")
    assert result["route"] == "structured_query"


def test_classify_normal_question_stays_rag():
    result = classify_route("what is the PTO policy")
    assert result["route"] == "production_rag"


def test_classify_research_keyword_falls_back():
    result = classify_route("compare market strategy vs competitors")
    assert result["route"] == "production_rag"
    assert result["intended_route"] == "deep_research"
    assert "not yet implemented" in result.get("route_fallback_reason", "")


# ── Graph test ──

@pytest.mark.asyncio
async def test_structured_query_graph_basic_flow():
    from rag_app.graphs.structured_query import build_structured_query_graph

    engine = SQLiteQueryEngine()
    engine.setup_schema("CREATE TABLE users (id INTEGER, name TEXT)")
    engine.setup_schema("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")

    model = FakeSQLModel(
        sql="SELECT COUNT(*) FROM users",
        answer="There are 2 users in the database."
    )
    graph = build_structured_query_graph(model, engine)

    state = {"question": "how many users are there", "route": "structured_query"}
    result = await graph.ainvoke(state)

    assert "answer" in result
    assert "2" in result["answer"]
    assert result.get("result") is not None
    assert result["result"].row_count == 1
    assert result["result"].rows[0][0] == 2


@pytest.mark.asyncio
async def test_table_allowlist_blocks_restricted_table():
    engine = SQLiteQueryEngine(table_allowlist=["users"])
    engine.setup_schema("CREATE TABLE users (id INTEGER); CREATE TABLE secrets (id INTEGER)")
    engine.setup_schema("INSERT INTO users VALUES (1); INSERT INTO secrets VALUES (99)")

    # Allowed table
    result = await engine.execute_readonly(StructuredQuery(sql="SELECT * FROM users"))
    assert result.row_count == 1

    # Blocked table
    with pytest.raises(ValueError, match="not in allowlist"):
        await engine.execute_readonly(StructuredQuery(sql="SELECT * FROM secrets"))


@pytest.mark.asyncio
async def test_default_limit_applied():
    engine = SQLiteQueryEngine(default_limit=5)
    engine.setup_schema("CREATE TABLE t (id INTEGER)")
    for i in range(20):
        engine.setup_schema(f"INSERT INTO t VALUES ({i})")

    result = await engine.execute_readonly(StructuredQuery(sql="SELECT * FROM t"))
    assert result.row_count == 5  # Limited to 5
