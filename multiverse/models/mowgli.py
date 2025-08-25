import argparse
import os
import json
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import mowgli
from .base import ModelFactory
from ..config import load_config
from ..logging_utils import get_logger, setup_logging
from ..utils import get_device
from ..data_utils import load_datasets, dataset_select

logger = get_logger(__name__)

class MowgliModel(ModelFactory):
    """Mowgli model implementation."""

    def __init__(self, dataset, dataset_name, config_path: str, is_gridsearch=False):
        logger.info("Initializing Mowgli Model")

        super().__init__(dataset, dataset_name, config_path=config_path,
                         model_name="mowgli", is_gridsearch=is_gridsearch)

        if self.model_name not in self.model_params:
            raise ValueError(f"'{self.model_name}' configuration not found in the model parameters.")

        mowgli_params = self.model_params.get(self.model_name)

        self.device = mowgli_params.get("device")
        self.torch_device = 'cpu'
        self.latent_dimensions = mowgli_params.get("latent_dimensions")
        self.optimizer = mowgli_params.get("optimizer")
        self.learning_rate = mowgli_params.get("learning_rate")
        self.inner_tolerance = mowgli_params.get("tol_inner")
        self.max_inner_iteration = mowgli_params.get("max_iter_inner")
        self.umap_color_type = mowgli_params.get("umap_color_type")
        self.umap_random_state = mowgli_params.get("umap_random_state")
        self.loss = 0
        self.model = mowgli.models.MowgliModel(latent_dim=self.latent_dimensions)
        self.torch_device = get_device(self.device)
        logger.info(f"Mowgli model initiated with {self.latent_dimensions} dimension.")

    def train(self):
        logger.info("Training Mowgli Model")
        try:
            self.model.train(
                self.dataset,
                device=self.torch_device,
                optim_name=self.optimizer,
                lr=self.learning_rate,
                tol_inner=self.inner_tolerance,
                max_iter_inner=self.max_inner_iteration
            )
            self.dataset.obsm[self.latent_key] = self.dataset.obsm["W_OT"]
            self.loss = self.model.losses[-1]
            logger.info(f"Final training loss: {self.loss}")
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
        """Generate UMAP visualization using Mowgli embeddings for all modalities."""
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
        metrics = {}
        metrics = {}
        if hasattr(self, "loss"):
            logger.info(f"Optimal Transport Loss (Mowgli): {self.loss}")
            metrics["ot_loss"] = str(-self.loss)
        else:
            logger.warning("Loss not available in the model.")

        scib_metrics = super().evaluate_model(label_key=self.umap_color_type)
        metrics.update(scib_metrics)

        try:
            with open(self.metrics_filepath, "w") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {self.metrics_filepath}")
        except IOError as e:
            logger.error(f"Could not write metrics file to {self.metrics_filepath}: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Run Mowgli model")
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
    data_concat = dataset_select(datasets_dict=datasets, data_type="mudata")

    try:
        for dataset_name, data_dict in data_concat.items():
            # Instantiate and run model
            model = MowgliModel(
                dataset=data_dict,
                dataset_name=dataset_name,
                config_path=args.config_path,
            )
            logger.info(f"Running Mowgli model on dataset: {dataset_name}")
            # Run the model pipeline
            model.train()
            model.save_latent()
            model.umap()
            model.evaluate_model()

            logger.info(f"Mowgli model run for {dataset_name} completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during Mowgli model run: {e}")
        # Optionally, re-raise the exception to indicate failure to the container runner
        raise

if __name__ == "__main__":
    main()
