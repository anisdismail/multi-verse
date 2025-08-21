import argparse
import os
import json
import scanpy as sc
import anndata as ad
import muon as mu
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from ..config import load_config
from ..train import load_datasets, dataset_select

from .base import ModelFactory

class MOFA_Model(ModelFactory):
    """MOFA Model implementation."""

    def __init__(self, dataset: ad.AnnData, dataset_name, config_path: str, is_gridsearch=False):
        print("Initializing MOFA Model")

        super().__init__(dataset, dataset_name, config_path=config_path,
                         model_name="mofa", is_gridsearch=is_gridsearch)

        if self.model_name not in self.model_params:
            raise ValueError(f"'{self.model_name}' configuration not found in the model parameters.")

        mofa_params= self.model_params.get(self.model_name)

        self.device = mofa_params.get("device")
        self.n_iterations = mofa_params.get("n_iterations")
        self.umap_color_type = mofa_params.get("umap_color_type")
        self.torch_device = "cpu"
        self.n_factors = mofa_params.get("n_factors")
        self.umap_random_state = mofa_params.get("umap_random_state")

    def to(self):
        """
        Method to set GPU or CPU mode for MOFA+.
        """
        if self.device != "cpu":
            self.gpu_mode = True
        else:
            self.gpu_mode = False
        print(f"Switching to {self.gpu_mode} mode")
    
    def train(self):
        """
        Train the MOFA model.
        """
        print("Training MOFA+ Model")
        try:
            mu.tl.mofa(
                data=self.dataset, n_factors=self.n_factors, gpu_mode=self.gpu_mode
            )
            print("MOFA training completed.")

            # Debugging output
            # print(f"Keys in dataset.uns['mofa']: {self.dataset.uns.get('mofa', {}).keys()}")

            # Compute explained variance if not available
            if "explained_variance" in self.dataset.uns.get("mofa", {}):
                self.explained_variance = self.dataset.uns["mofa"]["explained_variance"]
                print(f"Explained variance per factor: {self.explained_variance}")
            else:
                # Manually calculate explained variance
                self.explained_variance = self._compute_explained_variance()
                print(
                    f"Computed explained variance per factor: {self.explained_variance}"
                )

            print(f"Total explained variance: {sum(self.explained_variance)}")
        except Exception as e:
            print(f"Error during training: {e}")
            raise
    
    def _compute_explained_variance(self):
        """
        Compute explained variance for MOFA factors.
        """
        try:
            factors = self.dataset.obsm["X_mofa"]  # Extract latent factors
            # print(f"Latent factors (X_mofa) shape: {factors.shape}")

            # Compute total variance from raw data across modalities
            total_variance = 0
            for modality in self.dataset.mod.values():
                if hasattr(modality.X, "toarray"):
                    modality_data = (
                        modality.X.toarray()
                    )  # Convert sparse to dense if needed
                else:
                    modality_data = modality.X
                total_variance += np.var(modality_data, axis=0).sum()

            # print(f"Total variance from all modalities: {total_variance}")

            # Variance explained by factors
            factor_variances = np.var(factors, axis=0)
            # print(f"Factor variances: {factor_variances}")

            explained_variance_ratio = factor_variances / total_variance
            # print(f"Explained variance ratio per factor: {explained_variance_ratio}")
            return explained_variance_ratio

        except Exception as e:
            print(f"Error computing explained variance: {e}")
            return []
    def evaluate_model(self):
        """
        Evaluate the trained MOFA+ model based on explained variance.
        """
        if hasattr(self, "explained_variance"):
            total_variance = sum(self.explained_variance)
            print(f"Total Explained Variance (MOFA+): {total_variance}")
            
            metrics = {"total_variance": total_variance}
            with open(
                    self.metrics_filepath,
                    "w",
                ) as f:
                    json.dump(metrics, f, indent=4)
        else:
            raise ValueError("Explained variance not available for MOFA+.")
    
    def save_latent(self):
        if self.latent_filepath is None:
            raise ValueError("latent_filepath is not set. Cannot save latent data.")
        try:
            print("Saving latent data")
            self.dataset.obs["batch"] = "batch_1"
            self.dataset.write(self.latent_filepath)
            print(f"MultiVI model for dataset {self.dataset_name} was saved as {self.latent_filepath}")
        except Exception as e:
            print(f"Error saving latent data: {e}")

    def umap(self):
        if self.umap_filename is None:
            raise ValueError("umap_filename is not set. Cannot save UMAP plot.")

        print(f"Generating UMAP with {self.model_name} embeddings for all modalities")
        try:
            sc.pp.neighbors(
                self.dataset, use_rep="X_mofa", random_state=self.umap_random_state
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
        except Exception as e:
            print(f"Error generating UMAP: {e}")

    
def main():
    parser = argparse.ArgumentParser(description="Run MultiVI model")
    parser.add_argument("--config_path", type=str, default="/app/config_alldatasets.json", help="Path to the configuration file")
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
            model = MOFA_Model(
                dataset=data_dict,
                dataset_name=dataset_name,
                config_path=args.config_path,
            )
            print(f"Running MOFA model on dataset: {dataset_name}")
            # Run the model pipeline
            model.to()
            model.train()
            model.save_latent()
            model.umap()
            model.evaluate_model()

            # Write success log
            with open(log_file, "w") as f:
                f.write("MOFA model run completed successfully.\n")

    except Exception as e:
        # Write error log
        with open(log_file, "w") as f:
            f.write(f"An error occurred: {e}\n")
        print(f"An error occurred: {e}")
        # Optionally, re-raise the exception to indicate failure to the container runner
        raise

if __name__ == "__main__":
    main()
