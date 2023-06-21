from src.llmbot import AppConfig, build_retrieval_chain


class QueryLLM:
    def __init__(self, persist_directory: str):
        self.qa_chain = build_retrieval_chain(
            config=AppConfig(persist_directory=persist_directory)
        )

    def do(self, event):
        resp = self.qa_chain({"question": event["question"]})
        event.update(resp)
        return event
