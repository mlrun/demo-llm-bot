import mlrun
from kfp import dsl
from mlrun.runtimes.pod import KubeResource


def apply_openai_env(project: mlrun.projects.MlrunProject, fn: KubeResource) -> None:
    fn.set_env_from_secret(
        "OPENAI_API_KEY", f"mlrun-project-secrets-{project.name}", "OPENAI_API_KEY"
    )
    fn.set_env_from_secret(
        "OPENAI_API_BASE", f"mlrun-project-secrets-{project.name}", "OPENAI_API_BASE"
    )


@dsl.pipeline(name="LLM Pipeline")
def pipeline(persist_directory: str, source_directory: str, urls_file: str):
    # Get our project object:
    project = mlrun.get_current_project()

    # Ingest and index documents in vector store
    ingest_docs_fn = project.get_function("ingest-documents")
    ingest_docs_fn.apply(mlrun.mount_v3io())
    apply_openai_env(project=project, fn=ingest_docs_fn)
    ingest_docs_run = project.run_function(
        ingest_docs_fn,
        params={
            "persist_directory": persist_directory,
            "source_directory": source_directory,
        },
    )

    # Ingest and index URLs in vector store
    ingest_urls_fn = project.get_function("ingest-urls")
    ingest_urls_fn.apply(mlrun.mount_v3io())
    apply_openai_env(project=project, fn=ingest_urls_fn)
    ingest_urls_run = project.run_function(
        ingest_urls_fn,
        params={"persist_directory": persist_directory, "urls_file": urls_file},
    ).after(ingest_docs_run)

    # Serve LLM
    serving_fn = project.get_function("serve-llm")
    serving_fn.apply(mlrun.mount_v3io())
    apply_openai_env(project=project, fn=serving_fn)
    graph = serving_fn.set_topology("flow", engine="async")
    graph.add_step(
        name="llm",
        class_name="src.serve_llm.QueryLLM",
        persist_directory=str(persist_directory),
        full_event=True,
    ).respond()

    project.deploy_function(serving_fn).after(ingest_urls_run)
