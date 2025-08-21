import argparse
import os
import json
import scanpy as sc
import anndata as ad
import scvi
import pandas as pd
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from ..config import load_config
from ..train import load_datasets, dataset_select
from ..logging_utils import get_logger
from ..utils import get_device

from .base import ModelFactory

logger = get_logger(__name__)

class TotalVIModel(ModelFactory):
    """TotalVI Model implementation."""

    def __init__(self, dataset: ad.AnnData, dataset_name, config_path: str, is_gridsearch=False):
        logger.info("Initializing TotalVI Model")

        super().__init__(dataset, dataset_name, config_path=config_path,
                         model_name="totalvi", is_gridsearch=is_gridsearch)

        if self.model_name not in self.model_params:
            raise ValueError(f"'{self.model_name}' configuration not found in the model parameters.")

        totalvi_params = self.model_params.get(self.model_name)

        self.device = totalvi_params.get("device")
        self.max_epochs = totalvi_params.get("max_epochs")
        self.learning_rate = totalvi_params.get("learning_rate")
        self.umap_color_type = totalvi_params.get("umap_color_type")
        self.torch_device = "cpu"
        self.latent_dimensions = totalvi_params.get("latent_dimensions")
        self.umap_random_state = totalvi_params.get("umap_random_state")
        self.torch_device = get_device(self.device)

        scvi.model.TOTALVI.setup_anndata(
            self.dataset,
            protein_expression_obsm_key="protein_expression",
            batch_key="batch"
        )

        self.model = scvi.model.TOTALVI(self.dataset)

    def train(self):
        logger.info("Training TotalVI Model")
        try:
            self.model.to(self.torch_device)
            self.model.train()
            self.dataset.obsm[self.latent_key] = self.model.get_latent_representation()
            logger.info(f"TotalVI training completed.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def save_latent(self):
        if self.latent_filepath is None:
            raise ValueError("latent_filepath is not set. Cannot save latent data.")
        try:
            logger.info("Saving latent data")
            self.dataset.obs["batch"] = "batch_1"
            self.dataset.write(self.latent_filepath)
            logger.info(f"TotalVI model for dataset {self.dataset_name} was saved as {self.latent_filepath}")
        except IOError as e:
            logger.error(f"Could not write latent file to {self.latent_filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving latent data: {e}")
            raise

    def umap(self):
        if self.umap_filename is None:
            raise ValueError("umap_filename is not set. Cannot save UMAP plot.")

        logger.info(f"Generating UMAP with {self.model_name} embeddings for all modalities")
        try:
            sc.pp.neighbors(
                self.dataset, use_rep=self.latent_key, random_state=self.umap_random_state
            )
            sc.tl.umap(self.dataset, random_state=self.umap_random_state)
            self.dataset.obsm[f"X_{self.model_name}_umap"] = self.dataset.obsm[
                "X_umap"
            ].copy()
            if self.umap_color_type in self.dataset.obs:
                sc.pl.umap(self.dataset, color=self.umap_color_type, show=False)
            else:
                logger.warning(
                    f"UMAP color key '{self.umap_color_type}' not found in .obs. Plotting without color."
                )
                sc.pl.umap(self.dataset, show=False)

            plt.savefig(self.umap_filename, bbox_inches="tight")
            plt.close()

            logger.info(
                f"UMAP plot for {self.model_name} {self.dataset_name} saved as {self.umap_filename}"
            )
        except Exception as e:
            logger.error(f"Error generating UMAP: {e}")
            raise

    def evaluate_model(self):
        super().evaluate_model(label_key=self.umap_color_type)

def main():
    parser = argparse.ArgumentParser(description="Run TotalVI model")
    parser.add_argument("--config_path", type=str, default="/app/config_alldatasets.json", help="Path to the configuration file")
    args = parser.parse_args()

    config = load_config(config_path=args.config_path)
    os.makedirs(config["output_dir"], exist_ok=True)

    # Data information from config file
    datasets = load_datasets(args.config_path)
    data_concat = dataset_select(datasets_dict=datasets, data_type="concatenate")

    try:
        for dataset_name, data_dict in data_concat.items():
            # Instantiate and run model
            model = TotalVIModel(
                dataset=data_dict,
                dataset_name=dataset_name,
                config_path=args.config_path,
            )
            logger.info(f"Running TotalVI model on dataset: {dataset_name}")
            # Run the model pipeline
            model.train()
            model.save_latent()
            model.umap()
            model.evaluate_model()

            logger.info(f"TotalVI model run for {dataset_name} completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during TotalVI model run: {e}")
        # Optionally, re-raise the exception to indicate failure to the container runner
        raise

if __name__ == "__main__":
    main()
