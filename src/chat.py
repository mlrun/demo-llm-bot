import gradio as gr
import requests
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_to_dict

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def query_llm(endpoint_url: str, message: str) -> str:
    resp = requests.post(
        url=endpoint_url,
        json={"question": message, "chat_history": messages_to_dict(memory.chat_memory.messages)},
        verify=False,
    )
    resp_json = resp.json()
    ai_message = resp_json["output"]
    memory.save_context({"input": message}, {"output": ai_message})
    return ai_message


def reset_memory() -> None:
    memory.clear()
    return None


with gr.Blocks(analytics_enabled=False, theme=gr.themes.Soft()) as chat:
    with gr.Row():
        endpoint_url = gr.Textbox(
            label="Model Endpoint",
            placeholder="Enter model endpoint for inferencing...",
        )
    with gr.Row():
        chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=6):
            message = gr.Textbox(label="Q:", placeholder="Type a question and Enter")
        with gr.Column(scale=3):
            clear = gr.Button("Clear")

    def respond(endpoint_url, message, chat_history):
        bot_message = query_llm(endpoint_url=endpoint_url, message=message)
        chat_history.append((message, bot_message))
        return "", chat_history

    message.submit(respond, [endpoint_url, message, chatbot], [message, chatbot])
    clear.click(reset_memory, None, chatbot, queue=False)
