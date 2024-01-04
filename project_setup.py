import os

import mlrun


def setup(project: mlrun.projects.MlrunProject) -> mlrun.projects.MlrunProject:
    source = project.get_param("source")
    secrets_file = project.get_param("secrets_file")
    image = project.get_param("image")

    # Create project secrets and also load secrets in local environment
    if secrets_file and os.path.exists(secrets_file):
        project.set_secrets(file_path=secrets_file)
        mlrun.set_env_from_file(secrets_file)

    # Set project git/archive source and enable pulling latest code at runtime
    if source:
        print(f"Project Source: {source}")
        project.set_source(source, pull_at_runtime=True)

        # Export project to zip if relevant
        if ".zip" in source:
            print(f"Exporting project as zip archive to {source}...")
            project.export(source)

    # Set MLRun functions
    project.set_function(
        name="ingest", func="src/ingest.py", kind="job", with_repo=True, image=image
    )

    project.set_function(
        name="serve-llm",
        func="src/serve_llm.py",
        kind="serving",
        image=image,
        with_repo=True,
    )

    # Set MLRun workflows
    project.set_workflow(name="main", workflow_path="src/ingest_and_deploy_workflow.py")

    # Save and return the project:
    project.save()
    return project
