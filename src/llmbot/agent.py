from langchain.agents import AgentType, Tool, initialize_agent

from .chains import (
    build_conversational_retrieval_chain,
    build_csv_chain,
    build_sql_database_chain,
)
from .config import AppConfig, setup_logging

logger = setup_logging()


def parse_agent_output(agent_resp: dict) -> str:
    if isinstance(agent_resp["output"], dict):
        ai_message = agent_resp["output"]["answer"]
    else:
        ai_message = agent_resp["output"]
    return ai_message


def build_agent(config: AppConfig):
    conversational_retrieval_chain = build_conversational_retrieval_chain(config=config)
    penguin_sql_database_chain = build_sql_database_chain(
        config=config,
        db_uri=f"sqlite:///{config.repo_dir}/data/sqlite/palmer_penguins.db",
    )
    iris_csv_chain = build_csv_chain(
        config=config, csv_path=f"{config.repo_dir}/data/csv/iris.csv"
    )

    tools = [
        Tool(
            name="MLOps",
            func=conversational_retrieval_chain.__call__,
            description="""
            Useful for when you need to answer questions about mlops, mlrun,
            iguazio, machine learning, data science, or other related topics.
            """,
            return_direct=True,
        ),
        Tool(
            name="Palmer penguins",
            func=penguin_sql_database_chain.run,
            description="""
            Useful for when you need to answer questions about Adelie,
            Gentoo, or Chinstrap penguins using a SQL database.
            """,
            return_direct=True,
        ),
        Tool(
            name="Iris",
            func=iris_csv_chain.run,
            description="""
            Useful for when you need to answer questions about iris
            flowers. Includes petal sizes in cm per species.
            """,
            return_direct=True,
        ),
    ]

    return initialize_agent(
        tools=tools,
        llm=config.get_llm(),
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
    )
