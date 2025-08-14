import argparse
import os
import json
import scanpy as sc
import anndata as ad
import mudata as md
import mowgli
import torch

from .base import ModelFactory

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

        self.model = mowgli.models.MowgliModel(latent_dim=self.latent_dimensions)
        print(f"Mowgli model initiated with {self.latent_dimensions} dimension.")

        self.latent_filepath = None
        self.umap_filename = None

    def to(self):
        try:
            if self.device != 'cpu':
                if torch.cuda.is_available():
                    print("GPU available")
                    print(f"Moving Mowgli model to {self.device}")
                    self.torch_device = torch.device(self.device)
                    print(f"Mowgli model successfully moved to {self.device}")
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
        if self.umap_filename is None:
            raise ValueError("umap_filename is not set. Cannot save UMAP plot.")
        print("Generating UMAP plot")
        try:
            # Create a temporary AnnData object for UMAP generation
            adata_for_umap = ad.AnnData(self.dataset.obsm["X_mowgli"], obs=self.dataset.obs)
            sc.pp.neighbors(adata_for_umap, use_rep="X", random_state=1)
            sc.tl.umap(adata_for_umap, random_state=1)

            sc.settings.figdir = os.path.dirname(self.umap_filename)
            filename = os.path.basename(self.umap_filename)

            if self.umap_color_type in adata_for_umap.obs:
                sc.pl.umap(adata_for_umap, color=self.umap_color_type, save=filename)
            else:
                sc.pl.umap(adata_for_umap, save=filename)

            print(f"A UMAP plot for Mowgli model was successfully generated and saved as umap{filename}")
        except Exception as e:
            print(f"Error generating UMAP: {e}")

    def evaluate_model(self):
        if hasattr(self, "loss"):
            print(f"Optimal Transport Loss (Mowgli): {self.loss}")
            return -self.loss
        else:
            raise ValueError("Loss not available for Mowgli.")

def main():
    parser = argparse.ArgumentParser(description="Run Mowgli model")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing data.h5mu")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save results")
    parser.add_argument("--config_path", type=str, default="/app/config_alldatasets.json", help="Path to the configuration file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "log.txt")

    try:
        # Mowgli expects a MuData object, typically from a .h5mu file
        input_file = os.path.join(args.input_dir, "data.h5mu")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        mdata = md.read(input_file)

        model = Mowgli_Model(dataset=mdata, dataset_name="mowgli_dataset", config_path=args.config_path)

        model.output_dir = args.output_dir
        model.latent_filepath = os.path.join(args.output_dir, "embeddings.h5ad")
        model.umap_filename = os.path.join(args.output_dir, "umap.png")

        model.train()
        model.save_latent()
        model.umap()

        score = model.evaluate_model()
        metrics = {"neg_ot_loss": score}
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        with open(log_file, "w") as f:
            f.write("Mowgli model run completed successfully.\n")

    except Exception as e:
        with open(log_file, "w") as f:
            f.write(f"An error occurred: {e}\n")
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
