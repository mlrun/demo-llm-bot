import glob
import os
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredURLLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus

from .config import AppConfig, setup_logging

logger = setup_logging()

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

# Document processing


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    logger.info(f"Loading documents from {source_dir}")
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files
    ]

    return [load_single_document(file_path) for file_path in filtered_files]


def process_documents(
    documents: List[Document],
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


def get_existing_milvus_documents(store: Milvus) -> set:
    if store.col:
        resp = store.col.query(expr="pk >= 0", output_fields=["source"])
        return {s["source"] for s in resp}
    else:
        return set()


def ingest_documents(config: AppConfig, source_directory: str):
    store = config.get_vector_store()
    logger.info(f"Using vectorstore {store}")

    documents = load_documents(
        source_dir=source_directory,
        ignored_files=get_existing_milvus_documents(store),
    )

    texts = process_documents(
        documents=documents,
        chunk_size=config.embeddings_chunk_size,
        chunk_overlap=config.embeddings_chunk_overlap,
    )
    if texts:
        logger.info("Creating embeddings. May take some minutes...")
        store.add_documents(texts)

    logger.info("Ingestion complete")


# URL processing
def filter_urls(new_urls: List[str], existing_urls: List[str]) -> List[str]:
    return list(set(new_urls) - set(existing_urls))


def load_urls_to_documents(urls: List[str]) -> List[Document]:
    loader = UnstructuredURLLoader(urls=urls, headers={"User-Agent": "Mozilla/5.0"})
    return loader.load()


def ingest_urls(config: AppConfig, urls: List[str]) -> None:
    store = config.get_vector_store()
    logger.info(f"Using vectorstore {store}")

    filtered_urls = filter_urls(
        new_urls=urls,
        existing_urls=get_existing_milvus_documents(store),
    )
    documents = load_urls_to_documents(urls=filtered_urls)
    texts = process_documents(
        documents=documents,
        chunk_size=config.embeddings_chunk_size,
        chunk_overlap=config.embeddings_chunk_overlap,
    )

    if texts:
        logger.info("Creating embeddings. May take some minutes...")
        store.add_documents(texts)

    logger.info("Ingestion complete")
