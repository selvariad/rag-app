# rag-app/rag_app/deps.py
from rag_app.config import AppConfig
from rag_core.capabilities.retriever import HybridRetriever
from rag_core.capabilities.indexer import ChromaIndexer
from rag_core.capabilities.embedder import LocalEmbedder, OpenAIEmbedder
from rag_core.capabilities.chat_model import LangChainChatModel
from rag_core.memory.conversation import ConversationStore


_config: AppConfig | None = None
_retriever = None
_indexer = None
_embedder = None
_model = None
_conversation_store = None
_checkpointer = None
_sql_engine = None


def init(config: AppConfig):
    global _config, _retriever, _indexer, _embedder, _model, _conversation_store, _checkpointer
    _config = config

    if config.embedding.provider == "openai":
        _embedder = OpenAIEmbedder(model=config.embedding.model, api_key=config.embedding.api_key)
    else:
        _embedder = LocalEmbedder(model_name=config.embedding.model)

    vs = config.vector_store
    if vs.backend == "qdrant":
        raise NotImplementedError(
            "Qdrant backend is not yet implemented in rag-core. "
            "Set vector_store.backend to 'chromadb' to use the default backend."
        )
    else:
        _indexer = ChromaIndexer(
            persist_dir=vs.chromadb.persist_dir,
            collection_name=vs.chromadb.collection_name,
        )
        _retriever = HybridRetriever(
            embedder=_embedder,
            persist_dir=vs.chromadb.persist_dir,
            collection_name=vs.chromadb.collection_name,
            use_reranker=config.reranker.enabled,
        )

    _model = LangChainChatModel(
        provider=config.llm.provider,
        model=config.llm.model,
        api_key=config.llm.api_key,
        base_url=config.llm.base_url or None,
    )
    _conversation_store = ConversationStore()

    from rag_core.capabilities.structured_query import SQLiteQueryEngine
    global _sql_engine
    _sql_engine = SQLiteQueryEngine(db_path=config.structured_query.db_path)
    if config.structured_query.ddl:
        _sql_engine.setup_schema(config.structured_query.ddl)


def get_config() -> AppConfig:
    assert _config is not None
    return _config


def get_retriever():
    assert _retriever is not None
    return _retriever


def get_indexer():
    assert _indexer is not None
    return _indexer


def get_embedder():
    assert _embedder is not None
    return _embedder


def get_model():
    assert _model is not None
    return _model


def set_model(model):
    global _model
    _model = model


def get_conversation_store():
    assert _conversation_store is not None
    return _conversation_store


def set_checkpointer(cp):
    global _checkpointer
    _checkpointer = cp


def get_checkpointer():
    return _checkpointer


def get_sql_engine():
    return _sql_engine
