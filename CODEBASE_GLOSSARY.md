# Codebase Glossary – Multi-verse

## High-Level Overview

Multi-verse is a Python package designed to facilitate the comparison of multimodal data integration methods, specifically MOFA, MOWGLI, MultiVI, and PCA. It is built using Python and leverages libraries such as Scanpy, MuData, and scvi-tools for data handling and modeling. The project is structured as a containerized application, where each model runs in its own Docker container. A central orchestrator is used to manage the containers, pass data, and collect results.

## Directory Tree

```
├── multiverse/
│   ├── __init__.py
│   ├── config.py         # Loads the main JSON configuration.
│   ├── dataloader.py     # Contains data loading and preprocessing logic.
│   ├── models/           # Contains individual, executable model scripts.
│   │   ├── __init__.py
│   │   ├── base.py       # Base class for models.
│   │   ├── pca.py        # PCA model script.
│   │   ├── multivi.py    # MultiVI model script.
│   │   └── mowgli.py     # Mowgli model script.
│   └── runner/           # Orchestrator logic.
│       ├── __init__.py
│       ├── cli.py        # Command-line interface for the orchestrator.
│       └── docker_runner.py # Logic for running Docker containers.
├── docker/               # Docker-related files.
│   ├── pca.Dockerfile
│   ├── multivi.Dockerfile
│   └── mowgli.Dockerfile
├── docs/                 # Documentation files.
│   ├── MODEL_CONTAINERS.md
│   └── RUNNER.md
├── Makefile              # Makefile for building and running.
└── README.md             # Project documentation.
```

## Glossary

### Orchestrator (`multiverse/runner/`)

---

**File**: `multiverse/runner/cli.py`
**Name**: `main`
**Description**: The main command-line interface for running the orchestrator.
**Inputs**:
- `--models` (list of strings): A list of models to run (e.g., `pca`, `multivi`).
- `--input` (string): Path to the input data directory.
- `--output` (string): Path to the output results directory.
**Outputs**: None. Prints progress to the console and orchestrates the creation of output files by the model containers.
**Dependencies**: `argparse`, `os`, `multiverse.runner.docker_runner`.
**Business Logic**:
- Parses command-line arguments.
- Iterates through the requested models.
- For each model, it creates a dedicated output directory and calls `run_model_container`.

---

**File**: `multiverse/runner/docker_runner.py`
**Name**: `run_model_container`
**Description**: Spins up a Docker container for a specific model, mounts data volumes, and monitors its execution.
**Inputs**:
- `model_name` (string): The name of the model to run (e.g., `pca`).
- `input_dir` (string): The local path to the input data directory.
- `output_dir` (string): The local path for the output results directory.
- `extra_args` (list, optional): Extra command-line arguments to pass to the container's entrypoint.
**Outputs**: None. The function streams container logs to the console.
**Dependencies**: `docker`, `os`.
**Business Logic**:
- Uses the Docker SDK for Python to interact with the Docker daemon.
- Maps the `model_name` to a specific Docker image name (e.g., `pca` -> `multiverse-pca`).
- Runs the container with the input directory mounted as read-only (`/data/input`) and the output directory mounted as read-write (`/data/output`).

### Models (`multiverse/models/`)

---

**File**: `multiverse/models/base.py`
**Name**: `ModelFactory`
**Description**: A base class that provides a common interface and shared functionality for all integration models.
**Notes**: This class is inherited by the specific model implementations.

---

**File**: `multiverse/models/pca.py` (and others like it)
**Name**: `main`
**Description**: The main entrypoint for a standalone model script, designed to be run inside a Docker container.
**Inputs**:
- `--input_dir` (string): The path to the input data directory inside the container (e.g., `/data/input`).
- `--output_dir` (string): The path to the output directory inside the container (e.g., `/data/output`).
- `--config_path` (string): The path to the JSON configuration file inside the container.
**Outputs**:
- `embeddings.h5ad`: The learned latent representation.
- `metrics.json`: A JSON file with evaluation metrics.
- `log.txt`: A log file for the run.
**Business Logic**:
- Parses command-line arguments.
- Loads data from the input directory.
- Instantiates the specific model class (e.g., `PCA_Model`).
- Runs the training pipeline.
- Saves the results to the output directory.
