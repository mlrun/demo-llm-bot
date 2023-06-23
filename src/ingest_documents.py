import mlrun

from src.llmbot import AppConfig
from src.llmbot.ingestion import ingest_documents


def handler(
    context: mlrun.MLClientCtx,
    persist_directory: str,
):
    config = AppConfig(persist_directory=persist_directory)

    context.logger.info(
        f"Starting ingestion from source directory {config.source_directory}..."
    )
    ingest_documents(config=config)

    context.logger.info(
        f"Ingestion complete and stored in persist directory {config.persist_directory}"
    )
