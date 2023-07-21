import mlrun

from src.llmbot import AppConfig
from src.llmbot.ingestion import ingest_documents


def handler(context: mlrun.MLClientCtx, source_directory: str):
    config = AppConfig()

    context.logger.info(
        f"Starting ingestion from source directory {source_directory}..."
    )
    ingest_documents(config=config, source_directory=source_directory)

    context.logger.info("Ingestion complete")
