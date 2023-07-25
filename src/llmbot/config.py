from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Set

from langchain.chat_models.base import BaseChatModel
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, BaseSettings, Field, PyObject

# LLM model config


class LLMModelConfig(BaseModel):
    model: str
    temperature: int
    model_class: PyObject

    def get_llm(self) -> BaseChatModel:
        return self.model_class(model=self.model, temperature=self.temperature)


class OpenAIModelConfig(LLMModelConfig):
    model: str = "gpt-3.5-turbo"
    temperature: int = 0
    model_class: PyObject = "langchain.chat_models.openai.ChatOpenAI"


# Embeddings model config


class EmbeddingsModelConfig(BaseModel):
    name: str
    chunk_size: int
    chunk_overlap: int
    embeddings_class: PyObject
    encode_kwargs: dict = Field(default_factory=dict)

    def get_embeddings(self) -> Embeddings:
        return self.embeddings_class(
            model_name=self.name, encode_kwargs=self.encode_kwargs
        )


class HFEmbeddingsModelConfig(EmbeddingsModelConfig):
    name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embeddings_class: PyObject = "langchain.embeddings.HuggingFaceEmbeddings"
    encode_kwargs: dict = {"batch_size": 32}


class HFMultiLingualEmbeddingsModelConfig(EmbeddingsModelConfig):
    name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embeddings_class: PyObject = "langchain.embeddings.HuggingFaceEmbeddings"
    encode_kwargs: dict = {"batch_size": 32}


# Vector store config


class VectorStoreConfig(ABC):
    @abstractmethod
    def __init__(self, embedding_function: Embeddings, kind: str, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_existing_documents(self) -> Set:
        raise NotImplementedError()

    @abstractmethod
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        raise NotImplementedError()


class InMemChromaConfig(VectorStoreConfig):
    def __init__(
        self,
        embedding_function: Embeddings,
        kind: str = "Chroma (In Memory)",
        persist_directory: str = "db",
    ):
        from chromadb.config import Settings
        from langchain.vectorstores import Chroma

        self.kind = kind
        self.store = Chroma(
            embedding_function=embedding_function,
            persist_directory=persist_directory,
            client_settings=Settings(
                persist_directory=persist_directory,
                chroma_db_impl="duckdb+parquet",
                anonymized_telemetry=False,
            ),
        )

    def get_existing_documents(self) -> Set:
        collection = self.store.get()
        return {metadata["source"] for metadata in collection["metadatas"]}

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        document_ids = self.store.add_documents(documents=documents, **kwargs)
        self.store.persist()
        return document_ids


class RestChromaConfig(VectorStoreConfig):
    def __init__(
        self,
        embedding_function: Embeddings,
        kind: str = "Chroma (REST)",
        host: str = "localhost",
        port: int = 8000,
    ):
        from chromadb.config import Settings
        from langchain.vectorstores import Chroma

        self.kind = kind
        self.store = Chroma(
            embedding_function=embedding_function,
            client_settings=Settings(
                chroma_api_impl="rest",
                chroma_server_host=host,
                chroma_server_http_port=str(port),
                anonymized_telemetry=False,
            ),
        )

    def get_existing_documents(self) -> Set:
        collection = self.store.get()
        return {metadata["source"] for metadata in collection["metadatas"]}

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        document_ids = self.store.add_documents(documents=documents, **kwargs)
        self.store.persist()
        return document_ids


class MilvusConfig(VectorStoreConfig):
    def __init__(
        self,
        embedding_function: Embeddings,
        kind: str = "Milvus",
        host: str = "milvus",
        port: int = 19530,
    ):
        from langchain.vectorstores import Milvus

        self.kind = kind
        self.store = Milvus(
            embedding_function=embedding_function,
            connection_args={"host": host, "port": str(port)},
        )

    def get_existing_documents(self) -> Set:
        if self.store.col:
            resp = self.store.col.query(expr="pk >= 0", output_fields=["source"])
            return {s["source"] for s in resp}
        else:
            return set()

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        document_ids = self.store.add_documents(documents=documents, **kwargs)
        return document_ids


# Main config


class AppConfig(BaseSettings):
    embeddings_model: EmbeddingsModelConfig = HFEmbeddingsModelConfig()
    llm_model: LLMModelConfig = OpenAIModelConfig()
    vector_store_class: VectorStoreConfig = MilvusConfig
    store: PyObject = None

    # Nuclio functions store their code in a specific directory
    repo_dir: str = "/opt/nuclio"

    MLRUN_DBPATH: str
    OPENAI_API_KEY: str
    OPENAI_API_BASE: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_or_create_vectorstore(self, **kwargs) -> VectorStoreConfig:
        if self.store:
            return self.store
        self.store = self.vector_store_class(
            embedding_function=self.embeddings_model.get_embeddings(), **kwargs
        )
        return self.store


# Logging config


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("llmbot")
