from langchain import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

from ..config import AppConfig, setup_logging

logger = setup_logging()


def build_sql_database_chain(config: AppConfig, db_uri: str):
    llm = config.llm_model.get_llm()
    db = SQLDatabase.from_uri(db_uri)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return create_sql_agent(
        llm=llm, toolkit=toolkit, verbose=True, handle_parsing_errors=True
    )
