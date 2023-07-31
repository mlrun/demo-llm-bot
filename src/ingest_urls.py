from pathlib import Path

import mlrun

from src.llmbot import AppConfig
from src.llmbot.ingestion import ingest_urls


def handler(context: mlrun.MLClientCtx, urls_file: str):
    config = AppConfig()
    config.get_or_create_vectorstore(host="localhost")

    # Split file of URLs separated by newlines into list for processing
    urls = Path(urls_file).read_text().splitlines()

    context.logger.info(f"Starting ingestion of {len(urls)} urls...")
    ingest_urls(config=config, urls=urls)

    context.logger.info("Ingestion complete")
