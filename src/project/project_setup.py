import importlib

import mlrun

IMAGE_REQUIREMENTS = [
    "mlrun==1.3.3",
    "langchain==0.0.200",
    "chromadb==0.3.26",
    "sentence-transformers==2.2.2",
]


def assert_build():
    for module_name in IMAGE_REQUIREMENTS:
        module_without_version = module_name.split("==")[0]
        module = importlib.import_module(module_without_version)
        print(module.__version__)


def create_and_set_project(
    git_source: str,
    name: str = "llmbot",
    default_image: str = None,
    user_project: bool = False,
    env_file: str = None,
):
    """
    Creating the project for this demo.
    :param git_source:              the git source of the project.
    :param name:                    project name
    :param default_image:           the default image of the project
    :param user_project:            whether to add username to the project name

    :returns: a fully prepared project for this demo.
    """
    if env_file:
        mlrun.set_env_from_file(env_file=env_file)

    # Get / Create a project from the MLRun DB:
    project = mlrun.get_or_create_project(name=name, context="./", user_project=user_project)

    # Set or build the default image:
    if project.default_image is None:
        if default_image is None:
            print("Building image for the demo:")
            image_builder = project.set_function(
                "src/project/project_setup.py",
                name="image-builder",
                handler="assert_build",
                kind="job",
                image="mlrun/mlrun",
                requirements=IMAGE_REQUIREMENTS,
            )
            assert image_builder.deploy()
            default_image = image_builder.spec.image
        project.set_default_image(default_image)

    # Set the project git source:
    project.set_source(git_source, pull_at_runtime=True)

    # Set the data collection function:
    project.set_function(
        name="ingest-documents",
        func="src/project/functions/ingest_documents.py",
        kind="job",
        handler="handler",
        with_repo=True,
    )

    # Save and return the project:
    project.save()
    return project
