from pathlib import Path

import mlrun

from src.llmbot import AppConfig
from src.llmbot.ingestion import ingest_documents, ingest_urls

config = AppConfig()


def urls_handler(context: mlrun.MLClientCtx, urls_file: str):
    # Split file of URLs separated by newlines into list for processing
    urls = Path(urls_file).read_text().splitlines()

    context.logger.info(f"Starting ingestion of {len(urls)} urls...")
    ingest_urls(config=config, urls=urls)

    context.logger.info("Ingestion complete")


def documents_handler(context: mlrun.MLClientCtx, source_directory: str):
    context.logger.info(
        f"Starting ingestion from source directory {source_directory}..."
    )
    ingest_documents(config=config, source_directory=source_directory)

    context.logger.info("Ingestion complete")
