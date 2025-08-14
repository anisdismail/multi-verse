import argparse
import os
import json
import scanpy as sc
import anndata as ad
import scvi
import pandas as pd
from sklearn.metrics import silhouette_score

from .base import ModelFactory

class MultiVI_Model(ModelFactory):
    """MultiVI Model implementation."""

    def __init__(self, dataset: ad.AnnData, dataset_name, config_path: str, is_gridsearch=False):
        print("Initializing MultiVI Model")

        super().__init__(dataset, dataset_name, config_path=config_path,
                         model_name="multivi", is_gridsearch=is_gridsearch)

        if self.model_name not in self.model_params:
            raise ValueError(f"'{self.model_name}' configuration not found in the model parameters.")

        multivi_params= self.model_params.get(self.model_name)

        self.device = multivi_params.get("device")
        self.max_epochs = multivi_params.get("max_epochs")
        self.learning_rate = multivi_params.get("learning_rate")
        self.latent_key = "X_multivi"
        self.umap_color_type = multivi_params.get("umap_color_type")

        if self.umap_color_type not in self.dataset.obs:
            print(f"Warning: '{self.umap_color_type}' not found in dataset. Defaulting to None for coloring.")
            self.umap_color_type = None

        self.latent_filepath = None
        self.umap_filename = None

        if "feature_types" in self.dataset.var.keys():
            try:
                self.dataset = self.dataset[:, self.dataset.var["feature_types"].argsort()].copy()
                if "Protein Expression" in self.dataset.var["feature_types"].unique():
                    protein_indices = self.dataset.var["feature_types"] == "protein expression"
                    protein_expression = self.dataset.X[:, protein_indices]
                    protein_names = self.dataset.var_names[protein_indices]
                    protein_expression_df = pd.DataFrame(protein_expression,
                                     index=self.dataset.obs_names,
                                     columns=protein_names)
                    self.dataset.obsm["protein_expression"] = protein_expression_df
                    scvi.model.MULTIVI.setup_anndata(self.dataset, protein_expression_obsm_key="protein_expression")
                else:
                    scvi.model.MULTIVI.setup_anndata(self.dataset, protein_expression_obsm_key=None)

                self.model = scvi.model.MULTIVI(self.dataset,
                                                n_genes=(self.dataset.var["feature_types"] == "Gene Expression").sum(),
                                                n_regions=(self.dataset.var["feature_types"] == "Peaks").sum(),
                                                )
            except Exception as e:
                print(f"Something is wrong in MultiVI initialization: {e}")
                raise
        else:
            raise ValueError("MultiVI initialization needs 'feature_types' in variable keys to setup genes (RNA-seq) and genomic regions (ATAC-seq)!")

    def to(self):
        try:
            if self.device !='cpu':
                print(f"Moving MultiVI model to {self.device}")
                self.model.to_device(self.device)
                print(f"Model successfully moved to {self.device}")
            else:
                self.model.to_device(self.device)
                print(f"Recommend to use GPU instead of {self.device}")
        except Exception as e:
            print(f"Invalid device '{self.device}' specified. Use 'cpu' or 'gpu'.")

    def train(self):
        print("Training MultiVI Model")
        try:
            self.to()
            self.model.train()
            self.dataset.obsm[self.latent_key] = self.model.get_latent_representation()
            print(f"Multivi training completed.")
        except Exception as e:
            print(f"Error during training: {e}")
            raise

    def save_latent(self):
        if self.latent_filepath is None:
            raise ValueError("latent_filepath is not set. Cannot save latent data.")
        print("Saving latent data")
        try:
            self.dataset.obsm[self.latent_key] = self.model.get_latent_representation()
            self.dataset.obs["batch"] = "batch_1"
            self.dataset.write(self.latent_filepath)
            print(f"MultiVI model for dataset {self.dataset_name} was saved as {self.latent_filepath}")
        except Exception as e:
            print(f"Error saving latent data: {e}")

    def umap(self):
        if self.umap_filename is None:
            raise ValueError("umap_filename is not set. Cannot save UMAP plot.")
        print("Generating UMAP plot")
        try:
            sc.settings.figdir = os.path.dirname(self.umap_filename)
            filename = os.path.basename(self.umap_filename)
            sc.pp.neighbors(self.dataset, use_rep=self.latent_key, random_state=1)
            sc.tl.umap(self.dataset, random_state=1)
            if self.umap_color_type in self.dataset.obs:
                sc.pl.umap(self.dataset, color=self.umap_color_type, save=filename)
            else:
                sc.pl.umap(self.dataset, save=filename)
            print(f"A UMAP plot for MultiVI model with dataset {self.dataset_name} was succesfully generated and saved as umap{filename}")
        except Exception as e:
            print(f"Error generating UMAP: {e}")

    def evaluate_model(self):
        if "X_multivi" in self.dataset.obsm:
            latent = self.dataset.obsm["X_multivi"]
            if self.umap_color_type and self.umap_color_type in self.dataset.obs:
                labels = self.dataset.obs[self.umap_color_type]
                silhouette = silhouette_score(latent, labels)
                print(f"Silhouette Score (MultiVI): {silhouette}")
                return silhouette
            else:
                print("Labels not found for clustering evaluation. Returning 0.0")
                return 0.0
        else:
            raise ValueError("Latent representation (X_multivi) not found.")

def main():
    parser = argparse.ArgumentParser(description="Run MultiVI model")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing data.h5ad")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save results")
    parser.add_argument("--config_path", type=str, default="/app/config_alldatasets.json", help="Path to the configuration file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "log.txt")

    try:
        input_file = os.path.join(args.input_dir, "data.h5ad")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        adata = sc.read_h5ad(input_file)

        model = MultiVI_Model(dataset=adata, dataset_name="multivi_dataset", config_path=args.config_path)

        model.output_dir = args.output_dir
        model.latent_filepath = os.path.join(args.output_dir, "embeddings.h5ad")
        model.umap_filename = os.path.join(args.output_dir, "umap.png")

        model.train()
        model.save_latent()
        model.umap()

        score = model.evaluate_model()
        metrics = {"silhouette_score": score}
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        with open(log_file, "w") as f:
            f.write("MultiVI model run completed successfully.\n")

    except Exception as e:
        with open(log_file, "w") as f:
            f.write(f"An error occurred: {e}\n")
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
