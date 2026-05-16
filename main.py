# rag-app/main.py
import uvicorn
from rag_app.config import load_config
from rag_app.app import create_app


def main():
    config = load_config("config.yaml")
    app = create_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
