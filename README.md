# Interactive Bot Demo using LLMs and MLRun

This repository demonstrates the usage of Language Models (LLMs) and MLRun to build an interactive chatbot using your own data for Retrieval Augmented Question Answering. The data will be ingested and indexed into a Vector Database to be queried by an LLM in real-time.

The project utilizes MLRun for orchestration/deployment, HuggingFace embeddings for indexing data, ChromaDB for the vector database, OpenAI's GPT-3.5 model for generating responses, Langchain to retrieve relevant data from the vector store and augment the response from the LLM, and Gradio for building an interactive frontend.


## Getting Started

To get started with the Interactive Bot Demo, follow the instructions below and then see the [tutorial.ipynb](tutorial.ipynb):


### Installation


1. Setup MLRun on docker compose using the [mlrun-setup](https://github.com/mlrun/mlrun-setup) utility. Be sure to include the `--jupyter` and `--milvus` flags as they will be used in this example. For example:
    ```bash
    python mlsetup.py docker --jupyter --milvus --tag 1.3.3
    ```

    *Note: You can add an optional parameter for the Jupyter image. If using an M1/M2 Mac, you should use an `arm64` comaptible image such as `jupyter/scipy-notebook:2023-06-01`. The corresponding final command will be `python mlsetup.py docker -j jupyter/scipy-notebook:2023-06-01 --milvus --tag 1.3.3`.*

1. Open Jupyter at http://localhost:8888

1. Clone [this repo](https://github.com/mlrun/demo-llm-bot) inside the Jupyter container

1. This project uses `conda` for environment managmeent and `poetry` for dependency management. To get started, setup the Python environment using the provided `Makefile`:
    ```bash
    cd demo-llm-bot
    make conda-env
    ```

    *Note: If this command times out, repeat `conda-env` until it successfully installs.*

1. Copy the `mlrun.env` file to another name (e.g. `openai.env`) and populate with the required environment variables.
    - `OPENAI_API_KEY`: Obtain an API key from OpenAI to access the GPT-3.5 model. You can find instructions on how to obtain an API key in the [OpenAI docs](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key).

    - `OPENAI_API_BASE`: If your application uses a different API base than the default, you can specify it here. Otherwise, leave as default.

1. Open [tutorial.ipynb](tutorial.ipynb) in Jupyter with the newly created `llmbot` kernel. Run the notebook to deploy the example.

## Overview

There are two main portions of this project:

### Index Data and Deploy LLM

The first step is to run a pipeline using MLRun responsible for:
1. Ingesting and indexing data into the vector database
1. Deploying a real-time model serving endpoint for the Langchain + ChromaDB application

This can be done with the following:

```python
from src import create_and_set_project

ENV_FILE = "mlrun.env" # update to your .env file

project = create_and_set_project(
    env_file=ENV_FILE,
    git_source="git://github.com/mlrun/demo-llm-bot#main"
)

project.run(
    name="main",
    arguments={
        "source_directory" : "data/mlrun_docs_md",
        "urls_file" : "data/urls/mlops_blogs.txt"
    },
    watch=True,
    dirty=True
)
```

This results in the following MLRun workflow:
![](images/workflow.png)

### Interactive Chat Application

Once the data has been indexed and the LLM application is running, the endpoint can be directly queried via POST request like so:

```python
serving_fn = project.get_function("serve-llm", sync=True)
serving_fn.invoke(path="/", body={"question" : "how I deploy ML models?", "chat_history" : []})
```

Additionally, it can be used in the provided interactive chat application. This application will answer questions in a chatbot format using the provided documents as context. The response from the LLM will also specify which document was used to craft the response. It can be deployed locally with the following:

```python
from src import chat

chat.launch(server_name="0.0.0.0", ssl_verify=False)
```
![](images/chat.png)

The model endpoint at the top can be filled in using this info:

```python
endpoint_url = serving_fn.get_url()
endpoint_url
```
