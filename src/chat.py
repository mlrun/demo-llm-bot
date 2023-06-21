import gradio as gr
import requests


def query_llm(endpoint_url: str, message: str) -> str:
    resp = requests.post(url=endpoint_url, json={"question": message}, verify=False)
    resp_json = resp.json()
    return f"""{resp_json['answer']} \nSources: \n{resp_json['sources']}"""


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
    clear.click(lambda: None, None, chatbot, queue=False)
