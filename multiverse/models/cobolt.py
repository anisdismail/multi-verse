import argparse
import argparse
import os
import json
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import torch
from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt

# We need to adjust the import path to be relative to the multiverse package
from .base import ModelFactory
from ..config import load_config
from ..train import load_datasets
from ..data_utils import fuse_mudata
from ..logging_utils import get_logger, setup_logging
from ..utils import get_device

logger = get_logger(__name__)

class CoboltModel(ModelFactory):
    """Cobolt model implementation."""

    def __init__(self, dataset, dataset_name, config_path: str, is_gridsearch=False):
        logger.info("Initializing Cobolt Model")

        super().__init__(
            dataset,
            dataset_name,
            config_path=config_path,
            model_name="cobolt",
            is_gridsearch=is_gridsearch,
        )

        if self.model_name not in self.model_params:
            raise ValueError(
                f"'{self.model_name}' configuration not found in the model parameters."
            )

        cobolt_params = self.model_params.get(self.model_name)

        self.device = cobolt_params.get("device")
        self.torch_device = "cpu"
        self.latent_dimensions = cobolt_params.get("latent_dimensions")
        self.umap_color_type = cobolt_params.get("umap_color_type")
        self.umap_random_state = cobolt_params.get("umap_random_state")
        self.learning_rate = cobolt_params.get("learning_rate")
        self.num_epochs = cobolt_params.get("num_epochs")
        self.loss = 0
        self.torch_device = get_device(self.device)
        # initialize dataset
        self.single_data_list = []
        for modality, adata in zip(self.dataset["modalities"], self.dataset["data"]):
            self.single_data_list.append(
                SingleData(
                    feature_name=modality,
                    dataset_name=self.dataset_name,
                    feature=adata.var_names.to_numpy(),
                    count=adata.X,
                    barcode=adata.obs_names.to_numpy(),
                )
            )
            
        self.multiomic_dataset = MultiomicDataset.from_singledata(*self.single_data_list)
        
        self.model = Cobolt(
            dataset=self.multiomic_dataset,
            n_latent=self.latent_dimensions,
            lr=self.learning_rate,
            device=self.torch_device,
        )

        logger.info(f"Cobolt model initiated with {self.latent_dimensions} dimension.")
        
        self.dataset = fuse_mudata(
            list_anndata=self.dataset["data"], list_modality=self.dataset["modalities"]
        )

    def train(self):
        logger.info("Training Cobolt Model")
        try:
            self.model.train(num_epochs=self.num_epochs)
            self.loss = self.model.history["loss"][-1]  # Get the last loss value
            #save the embedding of the cells with count data for both modalities (intersection)
            self.dataset.obsm[self.latent_key] = self.model.get_all_latent()[0][
                [
                    self.multiomic_dataset.get_comb_idx(
                        [True] * len(self.multiomic_dataset.omic)
                    )
                ]
            ].squeeze(0)
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def save_latent(self):
        if self.latent_filepath is None:
            raise ValueError("latent_filepath is not set. Cannot save latent data.")
        try:
            logger.info("Saving latent data")
            adata = ad.AnnData(self.dataset.obsm[self.latent_key], obs=self.dataset.obs)
            adata.write(self.latent_filepath)
            logger.info(f"Latent data saved to {self.latent_filepath}")
        except IOError as e:
            logger.error(f"Could not write latent file to {self.latent_filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving latent data: {e}")
            raise

    def umap(self):
        """Generate UMAP visualization using Cobolt embeddings for all modalities."""
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
            logger.error(f"An error occurred during UMAP generation: {e}")
            raise

    def evaluate_model(self):
        if hasattr(self, "loss"):
            logger.info(f"Cobolt Loss: {self.loss}")
            metrics = {"loss": self.loss}
            try:
                with open(
                    self.metrics_filepath,
                    "w",
                ) as f:
                    json.dump(metrics, f, indent=4)
            except IOError as e:
                logger.error(f"Could not write metrics file to {self.metrics_filepath}: {e}")
                raise
        else:
            raise ValueError("Loss not available in the model.")


def main():
    parser = argparse.ArgumentParser(description="Run Cobolt model")
    parser.add_argument(
        "--config_path",
        type=str,
        default="/app/config_alldatasets.json",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    config = load_config(config_path=args.config_path)
    os.makedirs(config["output_dir"], exist_ok=True)
    setup_logging(config["output_dir"])

    # Data information from config file
    datasets = load_datasets(args.config_path)

    try:
        for dataset_name, data_dict in datasets.items():
            # Instantiate and run model
            model = CoboltModel(
                dataset=data_dict,
                dataset_name=dataset_name,
                config_path=args.config_path,
            )
            logger.info(f"Running Cobolt model on dataset: {dataset_name}")
            # Run the model pipeline
            model.train()
            model.save_latent()
            model.umap()
            model.evaluate_model()

            logger.info(f"Cobolt model run for {dataset_name} completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during Cobolt model run: {e}")
        # Optionally, re-raise the exception to indicate failure to the container runner
        raise


if __name__ == "__main__":
    main()
