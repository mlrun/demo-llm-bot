from src.llmbot import AppConfig, build_agent


def get_sources(agent_resp: dict) -> list:
    sources = []
    if "intermediate_steps" in agent_resp:
        for step in agent_resp["intermediate_steps"]:
            _, outputs = step
            if "sources" in outputs:
                sources.append(outputs["sources"])
    return sources


def format_response(answer: str, sources: str) -> str:
    sources = "\n".join(sources)
    return f"{answer} \n\nSOURCES:\n{sources}"


class QueryLLM:
    def __init__(self, persist_directory: str):
        self.agent = build_agent(config=AppConfig(persist_directory=persist_directory))

    def do(self, event):
        resp = self.agent(event["question"])
        resp.pop("chat_history")
        sources = get_sources(resp)
        if sources:
            formatted_response = format_response(answer=resp["output"], sources=sources)
            resp["output"] = formatted_response
        resp.pop("intermediate_steps")
        event.update(resp)
        return event
