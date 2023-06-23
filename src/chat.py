import gradio as gr
import requests


def query_llm(endpoint_url: str, message: str) -> str:
    resp = requests.post(url=endpoint_url, json={"question": message}, verify=False)
    resp_json = resp.json()
    return resp_json["output"]


def reset_memory(endpoint_url: str) -> None:
    resp = requests.get(url=f"{endpoint_url}/reset_memory", verify=False)
    resp_json = resp.json()
    print(resp_json["output"])
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
    clear.click(reset_memory, [endpoint_url], chatbot, queue=False)
