import numpy as np
import os
from ..config import load_config

class ModelFactory:
    """
    Other classes will inherit initial attributes of this class (config_file, dataset, dataset_name, ...)
    List of functions in each model is same as ModelFactory but how it works is different for each model
    """
    def __init__(self, dataset, dataset_name: str,
                 model_name:str = "", outdir="./outputs",
                 config_path: str="./config.json", is_gridsearch=False):
        self.model_params = load_config(config_path=config_path).get("model")
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.outdir = outdir
        self.model_name = model_name
        # Embeddings of the latent space
        self.latent = np.zeros((2,2))
        self.latent_filepath = None

        self.is_grid_search = is_gridsearch  # Flag for grid search runs
        self.base_filename = f"{self.model_name}_{self.dataset_name}"
        self.output_dir = os.path.join(self.outdir, f"{self.model_name}_output") # Default
        os.makedirs(self.output_dir, exist_ok=True)

    def update_output_dir(self):
        if self.is_grid_search:
            self.output_dir = os.path.join(self.outdir, "gridsearch_output")
            self.base_filename += "_gridsearch"
            self.latent_filepath = os.path.join(self.output_dir, f"{self.base_filename}.h5ad")
            self.umap_filename = os.path.join(self.output_dir, f"_{self.base_filename}.png")
        else:
            self.output_dir = os.path.join(self.outdir, f"{self.model_name}_output")

        os.makedirs(self.output_dir, exist_ok=True)

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
                print(f"Warning: Invalid parameter name '{key}'")

    def to(self):
        print("Setting device for model CPU or GPU.")

    def train(self):
        print("Training the model.")

    def save_latent(self):
        print("Saving latent representation of the model.")

    def load_latent(self):
        print("Loading the available latent representation.")

    def umap(self):
        print("Create umap for presentation.")

    def evaluate_model(self):
        print("Write evaluation metrics for thi specific model for gridsearch")
