import numpy as np
import os
import json
import pandas as pd
import scib_metrics
from ..config import load_config
from ..logging_utils import get_logger

logger = get_logger(__name__)

class ModelFactory:
    """
    Other classes will inherit initial attributes of this class (config_file, dataset, dataset_name, ...)
    List of functions in each model is same as ModelFactory but how it works is different for each model
    """

    def __init__(
        self,
        dataset,
        dataset_name: str,
        model_name: str = "",
        config_path: str = "./config.json",
        is_gridsearch=False,
    ):
        self.config_dict = load_config(config_path=config_path)
        self.model_params = self.config_dict.get("model")
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.output_dir = os.path.join(
            self.config_dict["output_dir"],
            self.dataset_name,
            self.model_name,
        )

        # Embeddings of the latent space
        self.latent_filepath = os.path.join(
            self.output_dir,
            "embeddings.h5ad",
        )
        self.umap_filename = os.path.join(
            self.output_dir,
            "umap.png",
        )
        self.metrics_filepath = os.path.join(
            self.output_dir,
            "metrics.json",
        )
        self.is_grid_search = is_gridsearch  # Flag for grid search runs
        os.makedirs(self.output_dir, exist_ok=True)
        self.latent_key = f"X_{self.model_name}"
        if self.model_name in self.model_params:
            model_specific_params = self.model_params.get(self.model_name)
            self.umap_color_type = model_specific_params.get("umap_color_type")


    """def update_output_dir(self):
        if self.is_grid_search:
            self.output_dir = os.path.join(self.outdir, "gridsearch_output")
            self.latent_filepath = os.path.join(
                self.output_dir,
                f"{self.model_name}_{self.dataset_name}_gridsearch.h5ad",
            )
            self.umap_filename = os.path.join(
                self.output_dir, f"{self.model_name}_{self.dataset_name}_gridsearch.png"
            )
        else:
            self.output_dir = os.path.join(self.outdir, f"{self.model_name}_output")

        os.makedirs(self.output_dir, exist_ok=True)
    """
    def update_parameters(self, **kwargs):
        """
        Updates the model parameters.
        Args:
            **kwargs: Keyword arguments with parameter names and their new values.
                     Example: update_parameters(n_factors=10, n_iteration=500)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Handle invalid parameter names if necessary
                logger.warning(f"Invalid parameter name '{key}'")

    def train(self):
        logger.info("Training the model.")

    def save_latent(self):
        logger.info("Saving latent representation of the model.")

    def load_latent(self):
        logger.info("Loading the available latent representation.")

    def umap(self):
        logger.info("Create umap for presentation.")

    def evaluate_model(self, batch_key="batch", label_key="cell_type"):
        """
        Evaluate the model using scib-metrics.
        """
        logger.info("Evaluating model with scib-metrics.")

        if self.latent_key not in self.dataset.obsm:
            raise ValueError(f"Latent representation '{self.latent_key}' not found in dataset.")

        if batch_key not in self.dataset.obs.columns:
            logger.warning(f"Batch key '{batch_key}' not found in .obs, creating a dummy batch.")
            self.dataset.obs[batch_key] = "batch_1"

        if label_key not in self.dataset.obs.columns:
            logger.warning(f"Label key '{label_key}' not found in .obs, skipping metrics that require it.")
            label_key = None

        bm = scib_metrics.benchmark.Benchmarker(
            self.dataset,
            batch_key=batch_key,
            label_key=label_key,
            embedding_obsm_keys=[self.latent_key],
        )

        bm.benchmark()
        results_df = bm.get_results(min_max_scale=False)
        if not results_df.empty:
            return results_df.to_dict('records')[0]
        return {}
