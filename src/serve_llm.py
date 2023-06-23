from langchain.memory import ConversationBufferMemory

from src.llmbot import AppConfig, build_agent


class QueryLLM:
    def __init__(self, persist_directory: str):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent = build_agent(
            config=AppConfig(persist_directory=persist_directory), memory=self.memory
        )

    def parse_agent_resp(self, agent_resp: dict) -> str:
        """Parse agent output and add to memory"""
        user_message = agent_resp["input"]
        if isinstance(agent_resp["output"], dict):
            ai_message = agent_resp["output"]["answer"]
        else:
            ai_message = agent_resp["output"]
        self.memory.chat_memory.add_user_message(user_message)
        self.memory.chat_memory.add_ai_message(ai_message)
        return ai_message

    def do(self, event):
        if "reset_memory" in event.path:
            self.memory.clear()
            print("Resetting memory...")
            event["output"] = "Memory reset successful"
        else:
            agent_resp = self.agent(
                {"input": event["question"], "chat_history": self.memory.chat_memory.messages}
            )
            event["output"] = self.parse_agent_resp(agent_resp=agent_resp)
        return event
