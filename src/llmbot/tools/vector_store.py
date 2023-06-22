from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Chroma

from ..config import AppConfig, setup_logging

logger = setup_logging()


def build_retrieval_qa_with_sources_chain(config: AppConfig):
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
