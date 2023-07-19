import mlrun

from src.llmbot import AppConfig
from src.llmbot.ingestion import ingest_documents


def handler(context: mlrun.MLClientCtx, persist_directory: str, source_directory: str):
    config = AppConfig(persist_directory=persist_directory)
    config.get_or_create_vectorstore(persist_directory=persist_directory)

    context.logger.info(
        f"Starting ingestion from source directory {source_directory}..."
    )
    ingest_documents(config=config, source_directory=source_directory)

    context.logger.info(
        f"Ingestion complete and stored in persist directory {config.persist_directory}"
    )
