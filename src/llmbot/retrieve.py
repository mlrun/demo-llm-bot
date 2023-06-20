import logging

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Chroma

from .config import AppConfig

logger = logging.getLogger(name="mlrun")


def build_retrieval_chain(config: AppConfig):
    # Embeddings function
    embeddings = config.embeddings_model.get_embeddings()

    # Vector store retriever
    db = Chroma(
        persist_directory=config.persist_directory,
        embedding_function=embeddings,
        client_settings=config.get_chroma_settings(),
    )
    retriever = db.as_retriever(search_kwargs={"k": config.retrieval_chain.k})

    # LLM
    llm = config.llm_model.get_llm()

    # QA chain with sources
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type=config.retrieval_chain.chain_type, retriever=retriever
    )


class QueryLLM:
    def __init__(self, persist_directory: str):
        self.qa_chain = build_retrieval_chain(
            config=AppConfig(persist_directory=persist_directory)
        )

    def do(self, event):
        resp = self.qa_chain({"question": event["question"]})
        event.update(resp)
        return event
