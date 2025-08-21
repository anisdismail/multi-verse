import argparse
import os
import json
import scanpy as sc
import anndata as ad
import mudata as md
import matplotlib.pyplot as plt
import mowgli
import torch

# We need to adjust the import path to be relative to the multiverse package
from .base import ModelFactory
from ..config import load_config
from ..train import load_datasets, dataset_select

class Mowgli_Model(ModelFactory):
    """Mowgli model implementation."""

    def __init__(self, dataset, dataset_name, config_path: str, is_gridsearch=False):
        print("Initializing Mowgli Model")

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
        print(f"Mowgli model initiated with {self.latent_dimensions} dimension.")

    def to(self):
        try:
            if self.device != 'cpu':
                if torch.cuda.is_available():
                    print("GPU available")
                    print(f"Moving Mowgli model to {self.device}")
                    self.torch_device = torch.device(self.device)
                else:
                    print("GPU cuda not available. Mowgli model will run with cpu")
            else:
                print("Mowgli model will run with cpu. Recommend to use GPU for computational efficiency.")
        except Exception as e:
            print(f"Invalid device '{self.device}' specified. Use 'cpu' or 'gpu'.")
            raise

    def train(self):
        print("Training Mowgli Model")
        try:
            self.model.train(
                self.dataset,
                device=self.torch_device,
                optim_name=self.optimizer,
                lr=self.learning_rate,
                tol_inner=self.inner_tolerance,
                max_iter_inner=self.max_inner_iteration
            )
            self.dataset.obsm["X_mowgli"] = self.dataset.obsm["W_OT"]
            self.loss = self.model.losses[-1]
            print(f"Final training loss: {self.loss}")
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_latent(self):
        if self.latent_filepath is None:
            raise ValueError("latent_filepath is not set. Cannot save latent data.")
        try:
            print("Saving latent data")
            adata = ad.AnnData(self.dataset.obsm['X_mowgli'], obs=self.dataset.obs)
            adata.write(self.latent_filepath)
            print(f"Latent data saved to {self.latent_filepath}")
        except Exception as e:
            print(f"Error saving latent data: {e}")

    def umap(self):
        """Generate UMAP visualization using Mowgli embeddings for all modalities."""
        if self.umap_filename is None:
            raise ValueError("umap_filename is not set. Cannot save UMAP plot.")

        print(f"Generating UMAP with {self.model_name} embeddings for all modalities")

        sc.pp.neighbors(
            self.dataset, use_rep="X_mowgli", random_state=self.umap_random_state
        )

        sc.tl.umap(self.dataset, random_state=self.umap_random_state)

        self.dataset.obsm[f"X_{self.model_name}_umap"] = self.dataset.obsm[
            "X_umap"
        ].copy()

        if self.umap_color_type in self.dataset.obs:
            sc.pl.umap(self.dataset, color=self.umap_color_type, show=False)
        else:
            print(
                f"Warning: UMAP color key '{self.umap_color_type}' not found in .obs. Plotting without color."
            )
            sc.pl.umap(self.dataset, show=False)

        plt.savefig(self.umap_filename, bbox_inches="tight")
        plt.close()

        print(
            f"UMAP plot for {self.model_name} {self.dataset_name} saved as {self.umap_filename}"
        )

    def evaluate_model(self):
        if hasattr(self, "loss"):
            print(f"Optimal Transport Loss (Mowgli): {self.loss}")
            metrics = {"ot_loss": -self.loss}
            with open(
                self.metrics_filepath,
                "w",
            ) as f:
                json.dump(metrics, f, indent=4)
        else:
            raise ValueError("Loss not available in the model.")


def main():
    parser = argparse.ArgumentParser(description="Run PCA model")
    parser.add_argument(
        "--config_path",
        type=str,
        default="/app/config_alldatasets.json",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    config_path = args.config_path
    config = load_config(config_path=config_path)

    # Define log file path
    log_file = os.path.join(config["output_dir"], "log.txt")

    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)

    # Data information from config file
    datasets = load_datasets(config_path)
    data_concat = dataset_select(datasets_dict=datasets, data_type="mudata")

    try:
        for dataset_name, data_dict in data_concat.items():
            # Instantiate and run model
            model = Mowgli_Model(
                dataset=data_dict,
                dataset_name=dataset_name,
                config_path=args.config_path,
            )
            print(f"Running Mowgli model on dataset: {dataset_name}")
            # Run the model pipeline
            model.to()
            model.train()
            model.save_latent()
            model.umap()
            model.evaluate_model()

            # Write success log
            with open(log_file, "w") as f:
                f.write("Mowgli model run completed successfully.\n")

    except Exception as e:
        # Write error log
        with open(log_file, "w") as f:
            f.write(f"An error occurred: {e}\n")
        print(f"An error occurred: {e}")
        # Optionally, re-raise the exception to indicate failure to the container runner
        raise

if __name__ == "__main__":
    main()
