# Codebase Glossary – Multi-verse

## High-Level Overview

Multi-verse is a Python package designed to facilitate the comparison of multimodal data integration methods, specifically MOFA, MOWGLI, MultiVI, and PCA. It is built using Python and leverages libraries such as Scanpy, MuData, and scvi-tools for data handling and modeling, and scIB for evaluation. The application is driven by a JSON configuration file that specifies the datasets, models, and parameters to use. The main workflow consists of data loading and preprocessing, model training, and evaluation of the results.

## Directory Tree

```
├── CoboltModel/    # A separate model (Cobolt) with its own Docker environment.
├── dataloader.py   # Handles data loading and preprocessing.
├── eval.py         # Evaluates model performance using scIB metrics.
├── main.py         # Entry point for the application.
├── model.py        # Defines the integration models (PCA, MOFA, MultiVI, Mowgli).
├── train.py        # Contains the training logic for the models.
├── utils.py        # Helper functions, including grid search implementation.
├── config.py       # Utility for loading the configuration file.
├── environment.yml # Conda environment specification.
├── README.md       # Project documentation.
└── outputs/        # Default directory for output files (latent embeddings, plots, results).
```

## Glossary

### `main.py`

---

**File**: `main.py`
**Name**: `main`
**Description**: The main entry point for the Multi-verse application.
**Inputs**:
- `sys.argv[1]` (string): The path to the JSON configuration file, provided as a command-line argument.
**Outputs**: None. The script prints progress to the console and saves outputs (models, plots, results) to files.
**Dependencies**:
- `sys`: To access command-line arguments.
- `train.Trainer`: To manage the model training process.
- `eval.Evaluator`: To evaluate the trained models.
- `config.load_config`: To load the JSON configuration.
- `utils.GridSearchRun`: To run the hyperparameter grid search.
**Dependents**: None (executed from the command line).
**Business Logic**:
- Parses the command-line arguments to get the path of the configuration file.
- Based on the `_run_user_params` flag in the config, it runs a standard training and evaluation pipeline.
- Based on the `_run_gridsearch` flag, it runs a hyperparameter grid search.
**Notes**: The script suppresses all warnings using `warnings.filterwarnings("ignore")`.
**Owner**: Yuxin Qiu, Thi Hanh Nguyen Ly, Zuzanna Olga Bednarska

### `config.py`

---

**File**: `config.py`
**Name**: `load_config`
**Description**: Loads the configuration from a JSON file and caches it for subsequent calls.
**Inputs**:
- `config_path` (string, optional): The path to the JSON configuration file. Defaults to `./config.json`.
**Outputs**:
- `dict`: A dictionary containing the hyperparameters and settings from the JSON file.
**Dependencies**:
- `json`: To parse the JSON file.
**Dependents**:
- `main.main`
- `train.Trainer`
- `eval.Evaluator`
- `dataloader.DataLoader`
- `dataloader.Preprocessing`
- `model.ModelFactory`
- `utils.GridSearchRun`
**Business Logic**:
- Uses a global `_config_cache` variable to implement a singleton pattern for the configuration object.
- If the cache is empty, it reads and parses the JSON file.
- All subsequent calls return the cached object, preventing multiple file reads.
**Notes**: This function ensures that the entire application uses a single, consistent configuration object.
**Owner**: Yuxin Qiu, Thi Hanh Nguyen Ly, Zuzanna Olga Bednarska

### `dataloader.py`

---

**File**: `dataloader.py`
**Name**: `DataLoader`
**Description**: Handles loading data from various file formats and preparing it for the models.
**Inputs**:
- `__init__`:
  - `file_path` (string): Path to the data file.
  - `modality` (string): The modality of the data (e.g., "rna", "atac").
  - `isProcessed` (bool): Flag indicating if the data is already preprocessed.
  - `annotation` (string): The key for cell type annotations in the data.
  - `config_path` (string): Path to the configuration file.
**Outputs**:
- `read_anndata`: `anndata.AnnData` object.
- `read_mudata`: `mudata.MuData` object.
- `fuse_mudata`: `mudata.MuData` object.
- `anndata_concatenate`: `anndata.AnnData` object.
- `preprocessing`: `anndata.AnnData` object.
**Dependencies**:
- `scanpy`, `anndata`, `mudata`, `muon`, `numpy`
- `config.load_config`
- `dataloader.Preprocessing`
**Dependents**:
- `train.Trainer`
**Business Logic**:
- `read_anndata` supports reading from `.csv`, `.tsv`, `.h5ad`, `.txt`, `.mtx`, `.h5mu`, and `.h5` files.
- `fuse_mudata` combines multiple `AnnData` objects into a `MuData` object, intersecting observations to ensure consistency.
- `anndata_concatenate` joins multiple `AnnData` objects along the variable axis.
- `preprocessing` acts as a pipeline, calling `read_anndata` and then, if the data is not marked as preprocessed, it instantiates `Preprocessing` to clean the data.
**Notes**: Contains hard-coded logic for handling specific dataset annotations and formats (e.g., Prostate data).
**Owner**: Yuxin Qiu, Thi Hanh Nguyen Ly, Zuzanna Olga Bednarska

---

**File**: `dataloader.py`
**Name**: `Preprocessing`
**Description**: Performs modality-specific data preprocessing.
**Inputs**:
- `__init__`:
  - `anndata` (`anndata.AnnData`): The AnnData object to be preprocessed.
  - `config_path` (string): Path to the configuration file to get preprocessing parameters.
**Outputs**:
- `rna_preprocessing`: A preprocessed `anndata.AnnData` object.
- `atac_preprocessing`: A preprocessed `anndata.AnnData` object.
- `adt_preprocessing`: A preprocessed `anndata.AnnData` object.
**Dependencies**:
- `scanpy`, `muon`
- `config.load_config`
**Dependents**:
- `dataloader.DataLoader`
**Business Logic**:
- Reads preprocessing parameters (e.g., filtering thresholds, number of variable genes) from the configuration file.
- `rna_preprocessing`: Calculates QC metrics, filters cells and genes, normalizes, log-transforms, and finds highly variable genes.
- `atac_preprocessing`: Similar to RNA, but with parameters tailored for ATAC-seq data.
- `adt_preprocessing`: Performs centered log-ratio (CLR) normalization for protein abundance data.
**Notes**: The preprocessing steps are based on standard Scanpy and Muon workflows.
**Owner**: Yuxin Qiu, Thi Hanh Nguyen Ly, Zuzanna Olga Bednarska

### `model.py`

---

**File**: `model.py`
**Name**: `ModelFactory`
**Description**: A base class that provides a common interface and shared functionality for all integration models.
**Inputs**:
- `__init__`:
  - `dataset` (`anndata.AnnData` or `mudata.MuData`): The input dataset.
  - `dataset_name` (string): The name of the dataset.
  - `model_name` (string): The name of the model.
  - `outdir` (string): The base output directory.
  - `config_path` (string): Path to the configuration file.
  - `is_gridsearch` (bool): Flag indicating if the run is part of a grid search.
**Outputs**: None. This is a base class and its methods are meant to be overridden.
**Dependencies**:
- `config.load_config`, `os`, `numpy`
**Dependents**:
- `PCA_Model`
- `MOFA_Model`
- `MultiVI_Model`
- `Mowgli_Model`
**Business Logic**:
- Initializes common attributes for all models.
- `update_output_dir` dynamically sets the output path based on whether it's a normal run or a grid search.
- Provides placeholder methods (`train`, `save_latent`, `umap`, etc.) that define the interface for all model classes.
**Notes**: This factory pattern allows `train.Trainer` to handle different models through a consistent API.
**Owner**: Yuxin Qiu, Thi Hanh Nguyen Ly, Zuzanna Olga Bednarska

---

**File**: `model.py`
**Name**: `PCA_Model`
**Description**: Implements Principal Component Analysis (PCA).
**Inputs**:
- `__init__`:
  - `dataset` (`anndata.AnnData`): The input dataset (concatenated modalities).
**Outputs**:
- `train`: Populates `self.dataset.obsm["X_pca"]` with the PCA embeddings.
- `evaluate_model`: Returns the total variance explained by the selected components.
**Dependencies**: `scanpy`
**Business Logic**:
- Wraps `scanpy.pp.pca` for training.
- The evaluation score for hyperparameter tuning is the total variance explained.
**Notes**: PCA is implemented as a baseline linear integration method. It does not support GPU computation.
**Owner**: Yuxin Qiu, Thi Hanh Nguyen Ly, Zuzanna Olga Bednarska

---

**File**: `model.py`
**Name**: `MOFA_Model`
**Description**: Implements the Multi-Omics Factor Analysis (MOFA+) model.
**Inputs**:
- `__init__`:
  - `dataset` (`mudata.MuData`): The input dataset with multiple modalities.
**Outputs**:
- `train`: Populates `self.dataset.obsm["X_mofa"]` with the MOFA factors.
- `evaluate_model`: Returns the total variance explained by the learned factors.
**Dependencies**: `muon`
**Business Logic**:
- Wraps `muon.tl.mofa` for training.
- The evaluation score for hyperparameter tuning is the total variance explained.
**Notes**: Supports GPU acceleration.
**Owner**: Yuxin Qiu, Thi Hanh Nguyen Ly, Zuzanna Olga Bednarska

---

**File**: `model.py`
**Name**: `MultiVI_Model`
**Description**: Implements the MultiVI (Multimodal Variational Integration) model.
**Inputs**:
- `__init__`:
  - `dataset` (`anndata.AnnData`): The input dataset (concatenated modalities).
**Outputs**:
- `train`: Populates `self.dataset.obsm["X_multivi"]` with the latent representation.
- `evaluate_model`: Returns the silhouette score of the latent space.
**Dependencies**: `scvi`, `pandas`, `sklearn`
**Business Logic**:
- Wraps the `scvi.model.MULTIVI` model for training.
- Requires a `feature_types` column in `dataset.var` to distinguish genes and peaks.
- The evaluation score for hyperparameter tuning is the silhouette score based on cell type labels.
**Notes**: Supports GPU acceleration.
**Owner**: Yuxin Qiu, Thi Hanh Nguyen Ly, Zuzanna Olga Bednarska

---

**File**: `model.py`
**Name**: `Mowgli_Model`
**Description**: Implements the Mowgli model for multimodal data integration.
**Inputs**:
- `__init__`:
  - `dataset` (`mudata.MuData`): The input dataset with multiple modalities.
**Outputs**:
- `train`: Populates `self.dataset.obsm["X_mowgli"]` with the integrated embeddings.
- `evaluate_model`: Returns the negative of the final Optimal Transport loss.
**Dependencies**: `mowgli`, `torch`
**Business Logic**:
- Wraps the `mowgli.models.MowgliModel` for training.
- The evaluation score for hyperparameter tuning is the negative of the OT loss, as a lower loss is better.
**Notes**: Supports GPU acceleration.
**Owner**: Yuxin Qiu, Thi Hanh Nguyen Ly, Zuzanna Olga Bednarska

### `train.py`

---

**File**: `train.py`
**Name**: `Trainer`
**Description**: Orchestrates the entire model training pipeline.
**Inputs**:
- `__init__`:
  - `config_path` (string): Path to the configuration file.
  - `hyperparams` (dict, optional): A dictionary of hyperparameters to override the config file, used for grid search.
**Outputs**: None. The class methods modify the state of model objects and save artifacts to disk.
**Dependencies**:
- `anndata`, `json`, `os`
- `model` (all model classes)
- `config.load_config`
- `dataloader.DataLoader`
**Dependents**:
- `main.main`
- `eval.Evaluator`
- `utils.GridSearchRun`
**Business Logic**:
- `load_datasets`: Iterates through datasets defined in the config and uses `DataLoader` to load and preprocess each modality.
- `dataset_select`: Prepares data for different model architectures. It creates concatenated `AnnData` objects (for PCA, MultiVI) and `MuData` objects (for MOFA+, Mowgli).
- `model_select`: Acts as a factory for models. Based on the config, it instantiates the required model objects for each dataset.
- `train`: The main execution method. It iterates through each selected model for each dataset, calls the model's `train()`, `save_latent()`, and `umap()` methods.
**Notes**: The `Trainer` class is the central coordinator for the training phase of the workflow.
**Owner**: Yuxin Qiu, Thi Hanh Nguyen Ly, Zuzanna Olga Bednarska

### `eval.py`

---

**File**: `eval.py`
**Name**: `Evaluator`
**Description**: Evaluates the performance of trained models using `scib` metrics.
**Inputs**:
- `__init__`:
  - `latent_dir` (string): The directory where latent representations are stored.
  - `output_file` (string): The path to save the final JSON results.
  - `trainer` (`train.Trainer`): An instance of the `Trainer` class.
**Outputs**: None. The class saves a `results.json` file with the evaluation metrics.
**Dependencies**:
- `os`, `json`, `numpy`, `pandas`, `scanpy`, `anndata`, `scib`
- `train.Trainer`
- `model` (all model classes)
- `config.load_config`
**Dependents**:
- `main.main`
**Business Logic**:
- `unintegrated_adata`: Retrieves the original, unprocessed dataset to serve as a baseline for integration metrics.
- `process_models`: Iterates through the latent representation files saved by each model. For each file, it loads the data and calls `calculate_metrics`.
- `calculate_metrics`: A wrapper for `scib.metrics.metrics`. It computes several standard metrics for evaluating single-cell integration, including Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and various silhouette scores.
**Notes**: The results for all models and datasets are compiled into a single JSON file for easy comparison.
**Owner**: Yuxin Qiu, Thi Hanh Nguyen Ly, Zuzanna Olga Bednarska

### `utils.py`

---

**File**: `utils.py`
**Name**: `GridSearchRun`
**Description**: Manages and executes a hyperparameter grid search for the models.
**Inputs**:
- `__init__`:
  - `config_path` (string): The path to the configuration file.
**Outputs**: None. The method prints a summary of the best parameters and scores to the console and saves the latent embeddings and UMAP plots for the best performing models.
**Dependencies**:
- `itertools.product`
- `train.Trainer`
- `config.load_config`
**Dependents**:
- `main.main`
**Business Logic**:
- `generate_param_combinations`: A static helper method that creates a Cartesian product of all hyperparameter values specified in the config's `grid_search_params` section for a model.
- `run`: If the `_run_gridsearch` flag is true in the config, this method iterates through each model and dataset. For each pair, it tries every combination of hyperparameters. It calls the model's `train()` and `evaluate_model()` methods and tracks the best score.
**Notes**: The results of the best model from the grid search are saved in a separate `gridsearch_output` directory to distinguish them from regular runs.
**Owner**: Yuxin Qiu, Thi Hanh Nguyen Ly, Zuzanna Olga Bednarska
