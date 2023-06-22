from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory

from .config import AppConfig, setup_logging
from .tools import build_math_chain, build_retrieval_qa_with_sources_chain

logger = setup_logging()


def build_agent(config: AppConfig):
    math_chain = build_math_chain(config=config)
    retrieval_qa_with_sources_chain = build_retrieval_qa_with_sources_chain(config=config)

    tools = [
        Tool(
            name="Vector store",
            func=retrieval_qa_with_sources_chain.run,
            description="""
            Useful for when you need to answer questions using documents from the vector store.
            """,
        ),
        Tool(
            name="Calculator",
            func=math_chain.run,
            description=f"""
            Useful when you need to do math operations or arithmetic.
            """,
        ),
    ]

    agent_kwargs = {
        "prefix": f"You are friendly virtual assistant. You are tasked to assist the user on questions related to their query. You have access to the following tools:"
    }

    return initialize_agent(
        tools=tools,
        llm=config.llm_model.get_llm(),
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=ConversationBufferMemory(memory_key="chat_history"),
        verbose=True,
        agent_kwargs=agent_kwargs,
    )
