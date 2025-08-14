import argparse
import os
import json
import scanpy as sc
import anndata as ad

# We need to adjust the import path to be relative to the multiverse package
from .base import ModelFactory

class PCA_Model(ModelFactory):
    """PCA implementation"""

    def __init__(self, dataset: ad.AnnData, dataset_name, config_path: str, is_gridsearch=False):
        """
        Initialize the PCA model with the specified parameters.
        Input data is AnnData object that was concatenated of multiple modality
        """
        print("Initializing PCA Model")

        super().__init__(dataset, dataset_name, config_path=config_path,
                         model_name="pca", is_gridsearch=is_gridsearch)

        # Check if model-specific params are present
        if self.model_name not in self.model_params:
            raise ValueError(f"'{self.model_name}' configuration not found in the model parameters.")

        pca_params = self.model_params.get(self.model_name)

        # PCA parameters from config file
        self.n_components = pca_params.get("n_components")
        self.device = pca_params.get("device")
        self.gpu_mode = False # Cpu default mode
        self.umap_random_state = pca_params.get("umap_random_state")
        self.umap_color_type = pca_params.get("umap_color_type")

        # Output paths will be set by the runner
        self.latent_filepath = None
        self.umap_filename = None

        print(f"PCA initialized with {self.dataset_name}, {self.n_components} components.")

    def to(self):
        """
        Method to set GPU or CPU mode.
        """
        if self.device != 'cpu':
            print("PCA does not support GPU. Using CPU instead.")
        else:
            print("Using CPU mode for PCA.")
        self.gpu_mode = False

    def train(self):
        """Perform PCA on all modalities concatenated."""
        print("Training PCA Model")

        if 'highly_variable' in self.dataset.var.keys():
            sc.pp.pca(self.dataset, n_comps=self.n_components, use_highly_variable=True)
        else:
            sc.pp.pca(self.dataset, n_comps=self.n_components, use_highly_variable=False)

        self.latent = self.dataset.obsm["X_pca"]
        self.variance_ratio = self.dataset.uns["pca"]["variance_ratio"]

        print(f"Training PCA completed with {self.n_components} components")
        print(f"Total variance explained: {sum(self.variance_ratio)}")

    def save_latent(self):
        """Save the PCA latent representations."""
        if self.latent_filepath is None:
            raise ValueError("latent_filepath is not set. Cannot save latent data.")

        print("Saving PCA latent embeddings")
        self.dataset.obs["batch"] = "batch_1"
        self.dataset.write(self.latent_filepath)
        print(f"Latent data saved to {self.latent_filepath}")

    def umap(self):
        """Generate UMAP visualization using PCA embeddings for all modalities."""
        if self.umap_filename is None:
            raise ValueError("umap_filename is not set. Cannot save UMAP plot.")

        print("Generating UMAP with PCA embeddings for all modalities")

        sc.pp.neighbors(self.dataset, use_rep="X_pca", random_state=self.umap_random_state)
        sc.tl.umap(self.dataset, random_state=self.umap_random_state)

        self.dataset.obsm["X_pca_umap"] = self.dataset.obsm["X_umap"].copy()

        sc.settings.figdir = os.path.dirname(self.umap_filename)
        filename = os.path.basename(self.umap_filename)

        # Check if color type exists, otherwise plot without color
        if self.umap_color_type in self.dataset.obs:
            sc.pl.umap(self.dataset, color=self.umap_color_type, save=filename)
        else:
            print(f"Warning: UMAP color key '{self.umap_color_type}' not found in .obs. Plotting without color.")
            sc.pl.umap(self.dataset, save=filename)

        print(f"UMAP plot for {self.model_name} {self.dataset_name} saved as umap{filename}")

    def evaluate_model(self):
        """
        Evaluate the trained PCA model based on variance explained.
        """
        print("Evaluating PCA model...")
        if hasattr(self, "variance_ratio"):
            total_variance = sum(self.variance_ratio)
            print(f"Total Variance Explained: {total_variance}")
            return total_variance
        else:
            raise ValueError("PCA variance ratio not available in the model.")

def main():
    parser = argparse.ArgumentParser(description="Run PCA model")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing data.h5ad")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save results")
    parser.add_argument("--config_path", type=str, default="/app/config_alldatasets.json", help="Path to the configuration file")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define log file path
    log_file = os.path.join(args.output_dir, "log.txt")

    try:
        # Load data
        input_file = os.path.join(args.input_dir, "data.h5ad")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        adata = sc.read_h5ad(input_file)

        # Instantiate and run model
        pca_model = PCA_Model(dataset=adata, dataset_name="pca_dataset", config_path=args.config_path)

        # Set output paths
        pca_model.output_dir = args.output_dir
        pca_model.latent_filepath = os.path.join(args.output_dir, "embeddings.h5ad")
        pca_model.umap_filename = os.path.join(args.output_dir, "umap.png")

        # Run the model pipeline
        pca_model.to()
        pca_model.train()
        pca_model.save_latent()
        pca_model.umap()

        # Save metrics
        metrics = {"total_variance": pca_model.evaluate_model()}
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        # Write success log
        with open(log_file, "w") as f:
            f.write("PCA model run completed successfully.\n")

    except Exception as e:
        # Write error log
        with open(log_file, "w") as f:
            f.write(f"An error occurred: {e}\n")
        print(f"An error occurred: {e}")
        # Optionally, re-raise the exception to indicate failure to the container runner
        raise

if __name__ == "__main__":
    main()
