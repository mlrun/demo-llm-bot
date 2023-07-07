from pathlib import Path

import mlrun

from src.llmbot import AppConfig
from src.llmbot.ingestion import ingest_urls


def handler(context: mlrun.MLClientCtx, persist_directory: str, urls_file: str):
    config = AppConfig(persist_directory=persist_directory)

    # Split file of URLs separated by newlines into list for processing
    urls = Path(urls_file).read_text().splitlines()

    context.logger.info(f"Starting ingestion of {len(urls)} urls...")
    ingest_urls(config=config, urls=urls)

    context.logger.info(
        f"Ingestion complete and stored in persist directory {config.persist_directory}"
    )
