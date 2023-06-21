import os

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
def pipeline(
    persist_directory: str,
):
    # Get our project object:
    project = mlrun.get_current_project()

    # Ingest and index data in vector store
    ingest_fn = project.get_function("ingest-documents")
    ingest_fn.apply(mlrun.mount_v3io())
    apply_openai_env(project=project, fn=ingest_fn)
    ingest_run = project.run_function(ingest_fn, params={"persist_directory": persist_directory})

    # Serve LLM
    serving_fn = project.get_function("serve-llm")
    serving_fn.apply(mlrun.mount_v3io())
    apply_openai_env(project=project, fn=serving_fn)
    graph = serving_fn.set_topology("flow", engine="async")
    graph.add_step(
        name="llm",
        class_name="src.serve_llm.QueryLLM",
        persist_directory=str(persist_directory),
    ).respond()

    project.deploy_function(serving_fn).after(ingest_run)
