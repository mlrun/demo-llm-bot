import os

import mlrun
from kfp import dsl


@dsl.pipeline(name="LLM Pipeline")
def kfpipeline(
    persist_directory: str,
):
    # Get our project object:
    project = mlrun.get_current_project()

    # Ingest and index data in vector store
    ingest_fn = project.get_function("ingest-documents")
    ingest_fn.apply(mlrun.mount_v3io())
    ingest_fn.set_envs(
        {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
        }
    )
    ingest_run = project.run_function(ingest_fn, params={"persist_directory": persist_directory})

    # Serve LLM
    serving_fn = project.get_function("serve-llm")
    serving_fn.apply(mlrun.mount_v3io())
    serving_fn.set_envs(
        {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
        }
    )
    graph = serving_fn.set_topology("flow", engine="async")
    graph.add_step(
        name="llm",
        class_name="src.project.workflows.ingest_and_serve.QueryLLM",
        persist_directory=str(persist_directory),
    ).respond()

    project.deploy_function(serving_fn).after(ingest_run)
