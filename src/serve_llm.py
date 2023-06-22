from src.llmbot import AppConfig, build_agent


class QueryLLM:
    def __init__(self, persist_directory: str):
        self.agent = build_agent(config=AppConfig(persist_directory=persist_directory))

    def do(self, event):
        resp = self.agent(event["question"])
        event.update(resp)
        return event
