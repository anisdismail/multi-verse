import argparse
import os
import json
import scanpy as sc
import anndata as ad
import mudata as md
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import pandas as pd
from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt

# We need to adjust the import path to be relative to the multiverse package
from .base import ModelFactory
from ..config import load_config
from ..train import load_datasets
from ..data_utils import fuse_mudata



class Cobolt_Wrapper(ModelFactory):
    """Cobolt model implementation."""

    def __init__(self, dataset, dataset_name, config_path: str, is_gridsearch=False):
        print("Initializing Cobolt Model")

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
        self.to()
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

        print(f"Cobolt model initiated with {self.latent_dimensions} dimension.")
        
        self.dataset = fuse_mudata(
            list_anndata=self.dataset["data"], list_modality=self.dataset["modalities"]
        )

    def to(self):
        try:
            if self.device != "cpu":
                if torch.cuda.is_available():
                    print("GPU available")
                    print(f"Moving Cobolt model to {self.device}")
                    self.torch_device = torch.device(self.device)
                else:
                    print("GPU cuda not available. Cobolt model will run with cpu")
            else:
                print(
                    "Cobolt model will run with cpu. Recommend to use GPU for computational efficiency."
                )
        except Exception as e:
            print(f"Invalid device '{self.device}' specified. Use 'cpu' or 'gpu'.")
            raise

    def train(self):
        print("Training Cobolt Model")
        try:
            self.model.train(num_epochs=self.num_epochs)
            self.loss = self.model.history["loss"][-1]  # Get the last loss value
            #save the embedding of the cells with count data for both modalities (intersection)
            self.dataset.obsm["X_cobolt"] = self.model.get_all_latent()[0][
                [
                    self.multiomic_dataset.get_comb_idx(
                        [True] * len(self.multiomic_dataset.omic)
                    )
                ]
            ].squeeze(0)
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_latent(self):
        if self.latent_filepath is None:
            raise ValueError("latent_filepath is not set. Cannot save latent data.")
        try:
            print("Saving latent data")
            adata = ad.AnnData(self.dataset.obsm["X_cobolt"], obs=self.dataset.obs)
            adata.write(self.latent_filepath)
            print(f"Latent data saved to {self.latent_filepath}")
        except Exception as e:
            print(f"Error saving latent data: {e}")

    def umap(self):
        """Generate UMAP visualization using Cobolt embeddings for all modalities."""
        if self.umap_filename is None:
            raise ValueError("umap_filename is not set. Cannot save UMAP plot.")

        print(f"Generating UMAP with {self.model_name} embeddings for all modalities")

        sc.pp.neighbors(
            self.dataset, use_rep="X_cobolt", random_state=self.umap_random_state
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
            print(f"Cobolt Loss: {self.loss}")
            metrics = {"loss": self.loss}
            with open(
                self.metrics_filepath,
                "w",
            ) as f:
                json.dump(metrics, f, indent=4)
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

    config_path = args.config_path
    config = load_config(config_path=config_path)

    # Define log file path
    log_file = os.path.join(config["output_dir"], "log.txt")

    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)

    # Data information from config file
    datasets = load_datasets(config_path)

    try:
        for dataset_name, data_dict in datasets.items():
            # Instantiate and run model
            model = Cobolt_Wrapper(
                dataset=data_dict,
                dataset_name=dataset_name,
                config_path=args.config_path,
            )
            print(f"Running Cobolt model on dataset: {dataset_name}")
            # Run the model pipeline
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
