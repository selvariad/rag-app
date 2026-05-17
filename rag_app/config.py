# rag-app/rag_app/config.py
import os
import re
from dataclasses import dataclass, field
import yaml


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""


@dataclass
class EmbeddingConfig:
    provider: str = "local"
    model: str = "BAAI/bge-m3"
    api_key: str = ""


@dataclass
class RerankerConfig:
    enabled: bool = False


@dataclass
class ChromaDBConfig:
    persist_dir: str = "./chroma_db"
    collection_name: str = "documents"


@dataclass
class QdrantConfig:
    url: str = "http://localhost:6333"
    collection_name: str = "documents"


@dataclass
class VectorStoreConfig:
    backend: str = "chromadb"  # chromadb | qdrant
    chromadb: ChromaDBConfig = field(default_factory=ChromaDBConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)


@dataclass
class RedisConfig:
    url: str = "redis://localhost:6379"


@dataclass
class StructuredQueryConfig:
    enabled: bool = True
    db_path: str = ":memory:"
    ddl: str = ""
    table_allowlist: list[str] = field(default_factory=list)  # empty = all allowed
    column_allowlist: dict[str, list[str]] = field(default_factory=dict)  # table → [cols]
    default_limit: int = 100


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class LangFuseConfig:
    enabled: bool = False
    public_key: str = ""
    secret_key: str = ""
    host: str = "http://localhost:3000"


@dataclass
class AppConfig:
    namespace: str = "default"
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    chromadb: ChromaDBConfig = field(default_factory=ChromaDBConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    structured_query: StructuredQueryConfig = field(default_factory=StructuredQueryConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    langfuse: LangFuseConfig = field(default_factory=LangFuseConfig)


_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_env(value: str) -> str:
    def _replace(m):
        return os.environ.get(m.group(1), "")
    return _VAR_RE.sub(_replace, value)


def load_config(path: str) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def _build(cls, data: dict):
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {}
        for k, v in data.items():
            if k in field_names:
                if isinstance(v, str):
                    v = _resolve_env(v)
                kwargs[k] = v
        return cls(**kwargs)

    llm = _build(LLMConfig, raw.get("llm", {}))
    embedding = _build(EmbeddingConfig, raw.get("embedding", {}))
    reranker = _build(RerankerConfig, raw.get("reranker", {}))
    chromadb = _build(ChromaDBConfig, raw.get("chromadb", {}))
    qdrant = _build(QdrantConfig, raw.get("vector_store", {}).get("qdrant", {}))
    vector_store = VectorStoreConfig(
        backend=raw.get("vector_store", {}).get("backend", "chromadb"),
        chromadb=chromadb,
        qdrant=qdrant,
    )
    redis = _build(RedisConfig, raw.get("redis", {}))
    structured = _build(StructuredQueryConfig, raw.get("structured_query", {}))
    server = _build(ServerConfig, raw.get("server", {}))
    langfuse = _build(LangFuseConfig, raw.get("langfuse", {}))

    return AppConfig(
        namespace=raw.get("namespace", "default"),
        llm=llm, embedding=embedding, reranker=reranker,
        chromadb=chromadb, vector_store=vector_store,
        structured_query=structured,
        redis=redis, server=server, langfuse=langfuse,
    )
