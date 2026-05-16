# rag-app/tests/test_config.py
from rag_app.config import load_config, AppConfig


def test_load_config_defaults():
    cfg = load_config("config.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.llm.provider in ("openai", "anthropic", "deepseek", "zhipu")
    assert cfg.embedding.provider in ("local", "openai")
    assert cfg.reranker.enabled is False


def test_app_config_namespace_default():
    cfg = AppConfig()
    assert cfg.namespace == "default"
