import importlib

import mlrun

IMAGE_REQUIREMENTS = [
    "mlrun==1.3.3",
    "langchain==0.0.200",
    "chromadb==0.3.26",
    "sentence-transformers==2.2.2",
    "openai==0.27.8",
]


def assert_build():
    for module_name in IMAGE_REQUIREMENTS:
        name, version = module_name.split("==")
        module = importlib.import_module(name)
        print(module.__version__)
        assert module.__version__ == version


def create_and_set_project(
    git_source: str,
    name: str = "llmbot",
    default_image: str = None,
    default_base_image: str = "mlrun/ml-models:1.3.3",
    user_project: bool = False,
    env_file: str = None,
    force_build: bool = False,
):
    """
    Creating the project for this demo.
    :param git_source:              the git source of the project.
    :param name:                    project name
    :param default_image:           the default image of the project
    :param user_project:            whether to add username to the project name

    :returns: a fully prepared project for this demo.
    """

    # Set MLRun DB endpoint
    if env_file:
        mlrun.set_env_from_file(env_file=env_file)

    # Get / Create a project from the MLRun DB:
    project = mlrun.get_or_create_project(name=name, context="./", user_project=user_project)

    # Set or build the default image:
    if force_build or project.default_image is None:
        if default_image is None:
            print("Building default project image...")
            image_builder = project.set_function(
                func="src/project_setup.py",
                name="image-builder",
                handler="assert_build",
                kind="job",
            )
            build_status = project.build_function(
                function=image_builder,
                base_image=default_base_image,
                requirements=IMAGE_REQUIREMENTS,
            )
            default_image = build_status.outputs["image"]
        project.set_default_image(default_image)

    # Set the project git source:
    project.set_source(git_source, pull_at_runtime=True)

    # Set MLRun functions
    project.set_function(
        name="ingest-documents",
        func="src/ingest_documents.py",
        kind="job",
        handler="handler",
        with_repo=True,
    )

    project.set_function(
        name="serve-llm",
        func="src/serve_llm.py",
        kind="serving",
        image=default_base_image,
        with_repo=True,
        requirements=IMAGE_REQUIREMENTS,
    )

    # Set MLRun workflows
    project.set_workflow(name="main", workflow_path="src/ingest_and_serve.py")

    # Save and return the project:
    project.save()
    return project
