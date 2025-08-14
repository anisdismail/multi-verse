import docker
import os

def run_model_container(model_name, input_dir, output_dir, extra_args=None):
    client = docker.from_env()

    image_map = {
        "pca": "multiverse-pca",
        "mofa": "multiverse-mofa",
        "multivi": "multiverse-multivi",
        "mowgli": "multiverse-mowgli",
        "cobolt": "multiverse-cobolt",
    }

    image = image_map[model_name]
    container = client.containers.run(
        image,
        command=extra_args or [],
        volumes={
            os.path.abspath(input_dir): {"bind": "/data/input", "mode": "ro"},
            os.path.abspath(output_dir): {"bind": "/data/output", "mode": "rw"},
        },
        detach=True,
        remove=True
    )

    for log in container.logs(stream=True):
        print(log.decode().strip())
