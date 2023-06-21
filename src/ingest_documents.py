import mlrun

from src.llmbot import AppConfig, ingest


def handler(
    context: mlrun.MLClientCtx,
    persist_directory: str,
):
    config = AppConfig(persist_directory=persist_directory)

    context.logger.info(
        f"Starting ingestion from source directory {config.source_directory}..."
    )
    ingest(config=config)

    context.logger.info(
        f"Ingestion complete and stored in persist directory {config.persist_directory}"
    )
