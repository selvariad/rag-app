# rag-app/rag_app/graphs/structured_query.py
from typing import TypedDict, NotRequired, Literal
from langgraph.graph import StateGraph, START, END

from rag_core.types import (
    Message, StructuredQuery, StructuredResult, KnowledgeRoute,
)
from rag_core.capabilities.chat_model import ChatModel
from rag_core.capabilities.structured_query import StructuredQueryEngine


class StructuredQueryState(TypedDict):
    question: str
    route: KnowledgeRoute
    sql: NotRequired[str]
    result: NotRequired[StructuredResult]
    answer: NotRequired[str]
    error: NotRequired[str]


def build_structured_query_graph(
    model: ChatModel, engine: StructuredQueryEngine, table_schema: str = ""
) -> StateGraph:
    graph = StateGraph(StructuredQueryState)

    graph.add_node("generate_sql", generate_sql_node(model, table_schema))
    graph.add_node("execute_readonly", execute_readonly_node(engine))
    graph.add_node("explain", explain_node(model))

    graph.add_edge(START, "generate_sql")
    graph.add_edge("generate_sql", "execute_readonly")
    graph.add_edge("execute_readonly", "explain")
    graph.add_edge("explain", END)

    return graph.compile()


def generate_sql_node(model: ChatModel, table_schema: str):
    async def _generate_sql(state: StructuredQueryState) -> dict:
        question = state["question"]
        schema_hint = table_schema or "No schema provided. Ask the user to describe their data."

        prompt = f"""You are a SQL expert. Generate a SELECT-only SQLite query for this question.

Database schema:
{schema_hint}

Question: {question}

Return ONLY the raw SQL query, no explanation, no markdown formatting."""

        resp = await model.invoke([Message(role="user", content=prompt)])
        sql = resp.content.strip()
        # Strip markdown code fences if present
        sql = sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()

        return {"sql": sql}
    return _generate_sql


def execute_readonly_node(engine: StructuredQueryEngine):
    async def _execute(state: StructuredQueryState) -> dict:
        sql = state.get("sql", "")
        if not sql:
            return {"error": "No SQL generated", "result": StructuredResult(columns=[], rows=[], row_count=0)}

        try:
            result = await engine.execute_readonly(
                StructuredQuery(sql=sql, table_schema="")
            )
            return {"result": result}
        except Exception as e:
            return {"error": str(e), "result": StructuredResult(columns=[], rows=[], row_count=0)}
    return _execute


def explain_node(model: ChatModel):
    async def _explain(state: StructuredQueryState) -> dict:
        question = state["question"]
        sql = state.get("sql", "")
        result = state.get("result")
        error = state.get("error", "")

        if error:
            answer = f"SQL execution failed: {error}"
        elif result and result.row_count > 0:
            rows_preview = "\n".join(
                str(row) for row in result.rows[:20]
            )
            prompt = f"""You are a data analyst. Explain the SQL query results in natural language.

Question: {question}
SQL: {sql}
Columns: {result.columns}
Rows ({result.row_count} total, showing up to 20):
{rows_preview}

Give a clear, concise answer based on the data. Cite specific numbers."""
            resp = await model.invoke([Message(role="user", content=prompt)])
            answer = resp.content
        else:
            answer = "The query returned no results."

        return {"answer": answer}
    return _explain
