from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory

from .chains import build_conversational_retrieval_chain, build_math_chain
from .config import AppConfig, setup_logging

logger = setup_logging()


def build_agent(config: AppConfig, memory: ConversationBufferMemory):
    read_only_memory = ReadOnlySharedMemory(memory=memory)

    conversational_retrieval_chain = build_conversational_retrieval_chain(
        config=config, read_only_memory=read_only_memory
    )
    math_chain = build_math_chain(config=config)

    tools = [
        Tool(
            name="Llama",
            func=conversational_retrieval_chain.__call__,
            description="""
            Useful for when you need to answer questions about llamas.
            """,
            return_direct=True,
        ),
        Tool(
            name="Calculator",
            func=math_chain.run,
            description=f"""
            Useful when you need to do math operations or arithmetic.
            """,
        ),
    ]

    return initialize_agent(
        tools=tools,
        llm=config.llm_model.get_llm(),
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
    )
