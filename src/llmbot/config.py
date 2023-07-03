import logging
import os

from chromadb.config import Settings
from langchain.chat_models.base import BaseChatModel
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, BaseSettings, PyObject

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

    def get_embeddings(self) -> Embeddings:
        return self.embeddings_class(model_name=self.name)


class HFEmbeddingsModelConfig(EmbeddingsModelConfig):
    name: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embeddings_class: PyObject = "langchain.embeddings.HuggingFaceEmbeddings"


# Retrieval chain config


class RetrievalChainConfig(BaseModel):
    chain_type: str
    k: int


class RetrievalQAWithSourcesChainConfig(RetrievalChainConfig):
    chain_type: str = "stuff"
    k: int = 3


# Main config


class AppConfig(BaseSettings):
    persist_directory: str = "db"
    source_directory: str = "data/sample"
    chroma_db_impl: str = "duckdb+parquet"
    embeddings_model: EmbeddingsModelConfig = HFEmbeddingsModelConfig()
    llm_model: LLMModelConfig = OpenAIModelConfig()
    retrieval_chain: RetrievalChainConfig = RetrievalQAWithSourcesChainConfig()

    # Nuclio functions store their code in a specific directory
    repo_dir: str = (
        "/opt/nuclio" if os.getenv("NUCLIO_FUNCTION_INSTANCE") else os.getcwd()
    )

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


# Logging config


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("llmbot")
