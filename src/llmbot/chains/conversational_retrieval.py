from langchain.agents import Tool
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.vectorstores import Chroma

from ..config import AppConfig, setup_logging

logger = setup_logging()


def build_conversational_retrieval_chain(config: AppConfig):
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

    # Memory - placeholder to be passed in during inferencing
    memory = ReadOnlySharedMemory(
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )

    return ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT),
        combine_docs_chain=load_qa_with_sources_chain(
            llm=llm, chain_type=config.retrieval_chain.chain_type
        ),
        memory=memory,
        verbose=True,
    )
