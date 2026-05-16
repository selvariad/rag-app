# rag-app/rag_app/worker.py
from arq.worker import create_worker
from arq.connections import RedisSettings
from rag_app.tasks import ingest_document
from rag_app.config import load_config
from rag_app import deps


async def startup(ctx):
    config = load_config("config.yaml")
    deps.init(config)


async def shutdown(ctx):
    pass


def run_worker(redis_url: str = "redis://localhost:6379"):
    worker = create_worker(
        functions=[ingest_document],
        redis_settings=RedisSettings.from_dsn(redis_url),
        on_startup=startup,
        on_shutdown=shutdown,
    )
    worker.run()


if __name__ == "__main__":
    run_worker()
