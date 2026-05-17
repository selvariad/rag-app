# rag-app/rag_app/graph.py
import re
from typing import NotRequired, Literal, TypedDict
from langgraph.graph import StateGraph, START, END

from rag_core.types import (
    RetrievalQuery, RetrievedChunk, Message, QueryResult, DEFAULT_NAMESPACE,
    KnowledgeRoute,
)
from rag_core.capabilities.retriever import Retriever
from rag_core.capabilities.chat_model import ChatModel


class RAGState(TypedDict):
    question: str
    route: NotRequired[KnowledgeRoute]
    intended_route: NotRequired[KnowledgeRoute]
    route_fallback_reason: NotRequired[str]
    rewritten_query: NotRequired[str]
    retrieval_query: NotRequired[RetrievalQuery]
    chunks: NotRequired[list[RetrievedChunk]]
    messages: NotRequired[list[Message]]
    answer: NotRequired[str]
    grounded: NotRequired[bool]
    grounding_reason: NotRequired[str]
    regen_count: NotRequired[int]
    rewrite_count: NotRequired[int]


def build_rag_graph(retriever: Retriever, model: ChatModel, checkpointer=None) -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("classify_route", classify_route_node())
    graph.add_node("rewrite", rewrite_query_node(model))
    graph.add_node("retrieve", retrieve_node(retriever))
    graph.add_node("generate", generate_node(model))
    graph.add_node("check", check_node(model))

    graph.add_edge(START, "classify_route")
    graph.add_edge("classify_route", "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "check")
    graph.add_conditional_edges("check", route_after_check, {
        "regenerate": "generate",
        "rewrite": "rewrite",
        "end": END,
    })

    return graph.compile(checkpointer=checkpointer)


def classify_route(question: str) -> dict:
    """Standalone route classifier. Returns {route, intended_route, route_fallback_reason}.
    Call before building a graph so we can dispatch to the correct graph.
    """
    STRUCTURED = re.compile(
        r"\b(SQL|table|count\s+(of|the)|SUM|AVG|GROUP\s+BY|order|metric|dashboard|"
        r"how many|how much|total|average|percent|revenue|customers?|users?)\b",
        re.IGNORECASE,
    )
    DEEP_RESEARCH = re.compile(
        r"\b(research|compare|survey|market|strategy|report|analysis|trend|"
        r"pros and cons|versus|vs\.?)\b",
        re.IGNORECASE,
    )
    AGENTIC = re.compile(
        r"\b(debug|code|file path|function|class|error|traceback|stack trace|"
        r"multi.?hop|follow.?up|where is|find .* in)\b",
        re.IGNORECASE,
    )

    if STRUCTURED.search(question):
        return {"route": "structured_query", "intended_route": "structured_query"}
    if AGENTIC.search(question):
        return {"route": "production_rag", "intended_route": "agentic_retrieval",
                "route_fallback_reason": "agentic_retrieval not yet implemented — using RAG"}
    if DEEP_RESEARCH.search(question):
        return {"route": "production_rag", "intended_route": "deep_research",
                "route_fallback_reason": "deep_research not yet implemented — using RAG"}
    return {"route": "production_rag", "intended_route": "production_rag"}


def classify_route_node():
    """Heuristic route classifier based on query keywords.
    Returns intended_route (best path), route (actual, may fall back),
    and route_fallback_reason (explanation if fallback happens).
    """
    STRUCTURED = re.compile(
        r"\b(SQL|table|count\s+(of|the)|SUM|AVG|GROUP\s+BY|order|metric|dashboard|"
        r"how many|how much|total|average|percent|revenue|customers?|users?)\b",
        re.IGNORECASE,
    )
    DEEP_RESEARCH = re.compile(
        r"\b(research|compare|survey|market|strategy|report|analysis|trend|"
        r"pros and cons|versus|vs\.?)\b",
        re.IGNORECASE,
    )
    AGENTIC = re.compile(
        r"\b(debug|code|file path|function|class|error|traceback|stack trace|"
        r"multi.?hop|follow.?up|where is|find .* in)\b",
        re.IGNORECASE,
    )

    async def _classify(state: RAGState) -> dict:
        question = state["question"]
        intended: KnowledgeRoute = "production_rag"
        reason = ""

        if AGENTIC.search(question):
            intended = "agentic_retrieval"
            reason = "agentic_retrieval not yet implemented — using RAG"
        elif STRUCTURED.search(question):
            return {"route": "structured_query", "intended_route": "structured_query"}
        elif DEEP_RESEARCH.search(question):
            intended = "deep_research"
            reason = "deep_research not yet implemented — using RAG"

        return {
            "route": "production_rag",
            "intended_route": intended,
            "route_fallback_reason": reason,
        }
    return _classify


def rewrite_query_node(model: ChatModel):
    async def _rewrite(state: RAGState) -> dict:
        rewrite_count = state.get("rewrite_count", 0) + 1
        question = state["question"]
        retrieval_query = state.get("retrieval_query")
        messages = state.get("messages", [])

        if retrieval_query is None or rewrite_count == 1:
            q = retrieval_query or RetrievalQuery(text=question)
        else:
            history = "\n".join(
                f"{m.role}: {m.content}" for m in messages[-6:]
            ) if messages else "(no previous conversation)"

            prompt = (
                f"Previous conversation:\n{history}\n\n"
                f"The previous search for '{question}' returned irrelevant results. "
                f"Using the conversation context above, reformulate the query "
                f"to better find the information. Original: {question}"
            )
            resp = await model.invoke([Message(role="user", content=prompt)])
            rewritten_text = resp.content.strip()
            q = RetrievalQuery(
                text=rewritten_text,
                namespace=retrieval_query.namespace,
                filters=retrieval_query.filters,
                top_k=retrieval_query.top_k,
            )

        return {
            "rewritten_query": q.text,
            "retrieval_query": q,
            "rewrite_count": rewrite_count,
        }
    return _rewrite


def retrieve_node(retriever: Retriever):
    async def _retrieve(state: RAGState) -> dict:
        query = state.get("retrieval_query")
        if query is None:
            question = state.get("rewritten_query", state["question"])
            query = RetrievalQuery(text=question)
        chunks = await retriever.retrieve(query)
        return {"chunks": chunks}
    return _retrieve


def generate_node(model: ChatModel):
    async def _generate(state: RAGState) -> dict:
        regen_count = state.get("regen_count", 0) + 1
        chunks = state.get("chunks", [])
        question = state["question"]

        context = "\n\n".join(
            f"[Source {c.rank}] {c.content}" for c in chunks
        ) if chunks else "No relevant documents found."

        grounding_instruction = ""
        if regen_count > 1:
            grounding_instruction = (
                "CRITICAL: Your previous answer was not grounded in the retrieved documents. "
                "Only use information explicitly stated in the context below. "
                "Cite specific sources. If the context doesn't contain the answer, say so."
            )

        fallback_note = ""
        reason = state.get("route_fallback_reason", "")
        if reason:
            fallback_note = (
                f"Note: This question is better suited for a different capability "
                f"({reason}). Answer with RAG for now, but mention this limitation "
                f"briefly at the start of your answer.\n\n"
            )

        prompt = f"""You are an enterprise document assistant. Answer based ONLY on the provided context.

When you use information from the context, cite the source inline like [Source 1] or [Source 2].
If multiple sources support the same claim, cite all of them.
If the context doesn't contain the answer, say so clearly without fabricating.

{fallback_note}Context:
{context}

{grounding_instruction}

Question: {question}

Answer:"""

        resp = await model.invoke([Message(role="user", content=prompt)])
        return {
            "answer": resp.content,
            "regen_count": regen_count,
            "messages": [Message(role="assistant", content=resp.content)],
        }
    return _generate


def check_node(model: ChatModel):
    async def _check(state: RAGState) -> dict:
        answer = state.get("answer", "")
        chunks = state.get("chunks", [])
        question = state["question"]

        if not chunks:
            return {"grounded": True, "grounding_reason": "ok"}

        context = " ".join(c.content for c in chunks)
        prompt = f"""Verify whether the following answer is supported by the provided context.

Context: {context}

Question: {question}

Answer: {answer}

Is the answer fully grounded in the context? Reply with ONLY one word:
- "ok" if the answer is fully supported by the context
- "model_strayed" if the context contains relevant information but the answer didn't use it
- "chunks_irrelevant" if the context doesn't contain information relevant to the question"""

        resp = await model.invoke([Message(role="user", content=prompt)])
        reason = resp.content.strip().lower()

        if reason not in ("ok", "model_strayed", "chunks_irrelevant"):
            reason = "ok"

        return {
            "grounded": reason == "ok",
            "grounding_reason": reason,
        }
    return _check


def route_after_check(state: RAGState) -> Literal["regenerate", "rewrite", "end"]:
    grounded = state.get("grounded", True)
    reason = state.get("grounding_reason", "ok")
    regen_count = state.get("regen_count", 0)
    rewrite_count = state.get("rewrite_count", 0)

    if grounded:
        return "end"
    elif reason == "model_strayed" and regen_count < 3:
        return "regenerate"
    elif reason == "chunks_irrelevant" and rewrite_count < 2:
        return "rewrite"
    else:
        return "end"
