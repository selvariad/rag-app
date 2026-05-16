"""Standalone configuration tool — no server required.

Usage:
  python -m rag_app.config_cli show
  python -m rag_app.config_cli set llm.provider deepseek
  python -m rag_app.config_cli set llm.model deepseek-chat
  python -m rag_app.config_cli set llm.api_key sk-xxx
  python -m rag_app.config_cli set llm.base_url https://api.deepseek.com/anthropic
  python -m rag_app.config_cli set embedding.provider openai
  python -m rag_app.config_cli set embedding.model text-embedding-3-small
  python -m rag_app.config_cli set embedding.api_key sk-xxx
  python -m rag_app.config_cli set reranker.enabled true
  python -m rag_app.config_cli providers
"""
import sys
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _read() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f.read())


def _write(data: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def cmd_show():
    data = _read()
    llm = data.get("llm", {})
    emb = data.get("embedding", {})
    reranker = data.get("reranker", {})
    print(f"LLM Provider   : {llm.get('provider', '?')}")
    print(f"LLM Model      : {llm.get('model', '?')}")
    print(f"LLM API Key    : {'***' if llm.get('api_key', '') and not llm.get('api_key', '').startswith('$') else '(not set)'}")
    print(f"LLM Base URL   : {llm.get('base_url', '') or '(default)'}")
    print(f"Embed Provider : {emb.get('provider', '?')}")
    print(f"Embed Model    : {emb.get('model', '?')}")
    print(f"Embed API Key  : {'***' if emb.get('api_key', '') and not emb.get('api_key', '').startswith('$') else '(not set)'}")
    print(f"Reranker       : {'enabled' if reranker.get('enabled') else 'disabled'}")


def cmd_set(key: str, value: str):
    data = _read()
    parts = key.split(".")
    if len(parts) == 2:
        section, field = parts
        if section not in data:
            print(f"Unknown section: {section}")
            sys.exit(1)
        if field not in data[section]:
            print(f"Unknown field: {field} in section {section}")
            sys.exit(1)
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        data[section][field] = value
        _write(data)
        print(f"Set {key} = {'***' if 'api_key' in key else value}")
    else:
        print(f"Invalid key: {key}. Use format: section.field")
        sys.exit(1)


def cmd_providers():
    print("Available LLM providers:")
    for p, desc in [("openai", "OpenAI"), ("anthropic", "Anthropic (Claude)"),
                     ("deepseek", "DeepSeek"), ("zhipu", "Zhipu (GLM)")]:
        print(f"  {p:<12} {desc}")
    print()
    print("Available embedding providers:")
    for p, desc in [("local", "Local (BGE-M3)"), ("openai", "OpenAI")]:
        print(f"  {p:<12} {desc}")
    print()
    print("Examples:")
    print("  python -m rag_app.config_cli set llm.provider deepseek")
    print("  python -m rag_app.config_cli set llm.model deepseek-chat")
    print("  python -m rag_app.config_cli set llm.api_key sk-your-key-here")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "show":
        cmd_show()
    elif cmd == "set":
        if len(sys.argv) != 4:
            print("Usage: python -m rag_app.config_cli set <section.field> <value>")
            sys.exit(1)
        cmd_set(sys.argv[2], sys.argv[3])
    elif cmd == "providers":
        cmd_providers()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
