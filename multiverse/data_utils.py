import anndata as ad
import mudata as md
import muon as mu
import numpy as np
from typing import List

def fuse_mudata(list_anndata: List[ad.AnnData] = None, list_modality: List[str] = None) -> md.MuData:
    """
    Fusing paired anndata as MuData
    intersect_obs will be used if number of obs not equivalent
    Args:
        list_modality: A list of strings representing the modalities (e.g., ["rna", "atac", "adt"]).
        list_anndata: A list of AnnData objects corresponding to the modalities (e.g., [adata_rna, adata_atac, adata_adt]).
    Returns:s
        A mudata.MuData object    
    """
    if len(list_modality) != len(list_anndata):
        raise ValueError("Length of list_modality and list_anndata must be equal!")
    else:
        data_dict = {}
        for i, mod in enumerate(list_modality):
            data_dict[mod] = list_anndata[i]
            try:
                list_anndata[i].X = np.array(list_anndata[i].X.todense())
            except:
                pass

    data = mu.MuData(data_dict)
    mu.pp.intersect_obs(data)   # Make sure number of cells are the same for all modalities

    # Hard-code "cell_type" to avoid conflict
    if "cell_type" in data["rna"].obs.columns:
        data.obs["cell_type"] = data["rna"].obs["cell_type"]
    else:
        # If there is no 'cell_type' annotation in rna modality
        print("No annotation -> setting annotation as 'cell_type' = 0 to avoid conflicts.")
        num_obs = data.n_obs
        data.obs["cell_type"] = np.zeros(num_obs, dtype=int)

    return data

def anndata_concatenate(list_anndata: List[ad.AnnData] = None, list_modality: List[str] = None) -> ad.AnnData:
    """
    Args:
        list_modality: A list of strings representing the modalities (e.g., ["rna", "atac", "adt"]).
        list_anndata: A list of AnnData objects corresponding to the modalities (e.g., [adata_rna, adata_atac, adata_adt]).
    Returns:
        A AnnData object
    """
    mudata = fuse_mudata(list_anndata=list_anndata, list_modality=list_modality)
    list_ann = []
    for mod in list_modality:
        list_ann.append(mudata[mod])

    anndata = ad.concat(list_ann, axis="var", label="cell_type", merge="unique", uns_merge="unique")

    # Hard-code "cell_type" to avoid conflict (Should already be available when fuse_mudata is called above)
    num_obs = anndata.n_obs
    if "cell_type" in mudata.obs.columns:
        anndata.obs["cell_type"] = mudata.obs["cell_type"]
    else:
        # No annotation -> setting annotation as 'cell_type' = 0 to avoid conflicts.
        anndata.obs["cell_type"] = np.zeros(num_obs, dtype=int)
    
    anndata.obs["modality"] = np.zeros(num_obs, dtype=int) # Adding this to prevent error in multiVI model
    return anndata
