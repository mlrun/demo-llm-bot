import logging

from chromadb.config import Settings
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, BaseSettings, PyObject

logger = logging.getLogger(name="mlrun")


class LLMModel(BaseModel):
    name: str
    temperature: int


class OpenAIModel(LLMModel):
    name: str = "gpt-3.5-turbo"
    temperature: int = 0


class EmbeddingsModel(BaseModel):
    name: str
    chunk_size: int
    chunk_overlap: int
    embeddings_fn: PyObject

    def get_embedding_fn(self) -> Embeddings:
        return self.embeddings_fn(model_name=self.name)


class HFEmbeddingsModel(EmbeddingsModel):
    name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embeddings_fn: PyObject = "langchain.embeddings.HuggingFaceEmbeddings"


class AppConfig(BaseSettings):
    persist_directory: str = "db"
    source_directory: str = "data/sample"
    chroma_db_impl: str = "duckdb+parquet"
    embeddings_model: EmbeddingsModel = HFEmbeddingsModel()
    llm_model: LLMModel = OpenAIModel()

    MLRUN_DBPATH: str
    OPENAI_API_KEY: str
    OPENAI_API_BASE: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_chroma_settings(self) -> Settings:
        return Settings(
            chroma_db_impl=self.chroma_db_impl,
            persist_directory=self.persist_directory,
            anonymized_telemetry=False,
        )
