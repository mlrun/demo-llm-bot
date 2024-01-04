import mlrun
from kfp import dsl


@dsl.pipeline(name="LLM Pipeline")
def pipeline(source_directory: str, urls_file: str):
    # Get our project object:
    project = mlrun.get_current_project()

    # Ingest and index documents in vector store
    ingest_docs_run = project.run_function(
        "ingest",
        handler="documents_handler",
        params={
            "source_directory": source_directory,
        },
    )

    # Ingest and index URLs in vector store
    ingest_urls_run = project.run_function(
        "ingest",
        handler="urls_handler",
        params={"urls_file": urls_file},
    )

    # Serve LLM
    serving_fn = project.get_function("serve-llm")
    graph = serving_fn.set_topology("flow", engine="async")
    graph.add_step(
        name="llm",
        class_name="src.serve_llm.QueryLLM",
        full_event=True,
    ).respond()

    project.deploy_function(serving_fn).after(ingest_docs_run, ingest_urls_run)
