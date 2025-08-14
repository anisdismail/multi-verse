I want a multi-container architecture where each integration model runs in its own Docker container, and a main orchestrator can spin them up, pass data, and collect results.
We’ll extend the plan to support that while still following the safe migration rules.

Below is a step-by-step implementation plan specifically for introducing per-model Docker images with a central runner.

⸻

Phase 0 – Preparation (Docker-specific)
	1.	Create a docker/ directory at the repo root.
Inside, you will keep separate Dockerfiles for each model:

docker/
  pca.Dockerfile
  mofa.Dockerfile
  multivi.Dockerfile
  mowgli.Dockerfile
  cobolt.Dockerfile

You can keep requirements-<model>.txt or pyproject-<model>.toml alongside each if they differ.

	2.	List model dependencies.
From the glossary and existing scripts, note which packages each model needs:
	•	PCA → scanpy, anndata, numpy, scikit-learn
	•	MOFA → MOFA2, R runtime, rpy2, plus base Python deps
	•	MultiVI → scvi-tools, torch
	•	Mowgli → tensorflow or any other Mowgli deps
	•	Cobolt → whatever is in its existing Dockerfile
	3.	Create per-model environment files.
Example:

docker/requirements-pca.txt
docker/requirements-mofa.txt
...


	4.	Ensure test data is small enough to be passed between containers without huge overhead.

⸻

Phase 1 – Container Build Setup

1. Per-model Dockerfiles

For example, docker/pca.Dockerfile:

FROM python:3.11-slim

WORKDIR /app
COPY multiverse ./multiverse
COPY docker/requirements-pca.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "-m", "multiverse.models.pca"]

Do similarly for mofa, multivi, mowgli, and cobolt — each one should:
	•	Contain only the dependencies for that model.
	•	Copy only the necessary part of the codebase (multiverse package, configs, etc.).
	•	Have an entrypoint that runs training/inference for that model.

Decision criteria:
	•	If a model has huge deps, keep them isolated to avoid bloating others.
	•	If a model shares >90% of deps with another, you can build from a common base image.

⸻

2. Local Build Commands

From the repo root:

docker build -f docker/pca.Dockerfile -t multiverse-pca .
docker build -f docker/mofa.Dockerfile -t multiverse-mofa .
...


⸻

Phase 2 – Orchestrator Design

We will create a main runner in Python that:
	1.	Reads a config (e.g., runner_config.yaml) specifying which models to run, input data path, and output location.
	2.	Spins up the corresponding Docker containers using the Docker SDK for Python.
	3.	Mounts volumes for data exchange.
	4.	Waits for completion and collects results.

⸻

1. Orchestrator File Structure

multiverse/
  runner/
    __init__.py
    docker_runner.py   # spins up and monitors containers
    config.py          # reads runner_config.yaml
    io.py              # moves data in/out


⸻

2. docker_runner.py Implementation Plan
	1.	Install docker SDK:

pip install docker


	2.	Core function:

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

	3.	Verification step:
	•	After the container finishes, check output_dir for expected artifacts (embeddings.h5ad, metrics.json, etc.).
	•	If missing, log an error and stop.

⸻

Phase 3 – Data Exchange Format
	•	Use a shared .h5ad or .mudata file for input.
	•	Output directory must have:
	•	embeddings.h5ad
	•	metrics.json
	•	log.txt

This guarantees consistency across all model containers.

⸻

Phase 4 – Orchestrator CLI

Add a command-line tool:

import argparse
from multiverse.runner.docker_runner import run_model_container

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    for model in args.models:
        run_model_container(model, args.input, f"{args.output}/{model}")

Run it as:

python -m multiverse.runner.cli --models pca mofa multivi --input /path/to/data --output /path/to/results


⸻

Phase 5 – Testing the Multi-container Workflow
	1.	Build all images.
	2.	Run orchestrator with a dummy .h5ad file.
	3.	Check:
	•	All containers complete without error.
	•	Output directories contain the expected files.
	•	No cross-contamination between models.

⸻

Phase 6 – CI/CD Integration
	•	Add a CI matrix to build all model images separately.
	•	For smoke tests, run each container with a small fixture inside CI.

⸻

Phase 7 – Final Verification & Documentation
	1.	Document each model’s Docker build and run process in docs/MODEL_CONTAINERS.md.
	2.	Document orchestrator usage in docs/RUNNER.md.
	3.	Ensure make build-all builds all containers and make run-all runs orchestrator.

⸻
