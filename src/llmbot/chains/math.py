from langchain import LLMMathChain

from ..config import AppConfig, setup_logging

logger = setup_logging()


def build_math_chain(config: AppConfig):
    llm = config.llm_model.get_llm()
    return LLMMathChain.from_llm(llm=llm, verbose=True)
