from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from ..config import AppConfig, setup_logging

logger = setup_logging()


def filter_urls(new_urls: List[str], existing_urls: List[str]) -> List[str]:
    return list(set(new_urls) - set(existing_urls))


def load_urls_to_documents(urls: List[str]) -> List[Document]:
    loader = UnstructuredURLLoader(urls=urls, headers={"User-Agent": "Mozilla/5.0"})
    return loader.load()


def process_documents(
    documents: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Load documents and split in chunks
    """
    if not documents:
        logger.info("No new documents to load")
        return
    logger.info(f"Loaded {len(documents)} new documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    logger.info(
        f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)"
    )
    return texts


def ingest_urls(config: AppConfig, urls: List[str]) -> None:
    logger.info(f"Using vectorstore at {config.persist_directory}")
    db = Chroma(
        persist_directory=config.persist_directory,
        embedding_function=config.embeddings_model.get_embeddings(),
        client_settings=config.get_chroma_settings(),
    )
    collection = db.get()

    filtered_urls = filter_urls(
        new_urls=urls,
        existing_urls=[metadata["source"] for metadata in collection["metadatas"]],
    )
    documents = load_urls_to_documents(urls=filtered_urls)
    texts = process_documents(
        documents=documents,
        chunk_size=config.embeddings_model.chunk_size,
        chunk_overlap=config.embeddings_model.chunk_overlap,
    )

    if texts:
        logger.info("Creating embeddings. May take some minutes...")
        db.add_documents(texts)

    db.persist()
    db = None

    logger.info("Ingestion complete")
