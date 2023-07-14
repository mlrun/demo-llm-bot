import glob
import os
from multiprocessing import Pool
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
from langchain.vectorstores import Chroma
from tqdm import tqdm

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


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

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
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(
            total=len(filtered_files), desc="Loading new documents", ncols=80
        ) as pbar:
            for i, docs in enumerate(
                pool.imap_unordered(load_single_document, filtered_files)
            ):
                results.extend(docs)
                pbar.update()

    return results


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


def ingest_documents(config: AppConfig):
    logger.info(f"Using vectorstore at {config.persist_directory}")
    db = Chroma(
        persist_directory=config.persist_directory,
        embedding_function=config.embeddings_model.get_embeddings(),
        client_settings=config.get_chroma_settings(),
    )
    collection = db.get()

    documents = load_documents(
        source_dir=config.source_directory,
        ignored_files=[metadata["source"] for metadata in collection["metadatas"]],
    )

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


# URL processing
def filter_urls(new_urls: List[str], existing_urls: List[str]) -> List[str]:
    return list(set(new_urls) - set(existing_urls))


def load_urls_to_documents(urls: List[str]) -> List[Document]:
    loader = UnstructuredURLLoader(urls=urls, headers={"User-Agent": "Mozilla/5.0"})
    return loader.load()


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
