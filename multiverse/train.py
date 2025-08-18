from .dataloader import DataLoader
from .data_utils import fuse_mudata, anndata_concatenate
from .config import load_config
import os

def load_datasets(config_path):
    """
    Load all datasets specified in the configuration.
    Returns a dictionary where keys are dataset names, and values are the data objects.
    """
    config_dict = load_config(config_path)
    datasets = {}

    data_config = config_dict.get("data", {})
    dataset_names = [
        key
        for key, value in data_config.items()
        if isinstance(value, dict) and "data_path" in value
    ]

    for dataset_name in dataset_names:
        dataset_info = data_config[dataset_name]
        modality_list = [
            key
            for key, value in dataset_info.items()
            if isinstance(value, dict) and "file_name" in value
        ]  # modality (i.e.'rna') must be a dictionary
        dataset_path = dataset_info["data_path"]
        list_anndata = []
        # Check if data is loaded correctly
        if modality_list is not None:
            for modality in modality_list:
                modality_info = dataset_info[modality]
                file_path = os.path.join(dataset_path, modality_info["file_name"])
                is_preprocessed = modality_info["is_preprocessed"]
                annotation = modality_info["annotation"]
                ann_loader = DataLoader(
                    file_path=file_path,
                    modality=modality,
                    isProcessed=is_preprocessed,
                    annotation=annotation,
                    config_path=config_path,
                )
                ann = ann_loader.preprocessing()
                list_anndata.append(ann)
            datasets[dataset_name] = {"modalities": modality_list, "data": list_anndata}
        else:
            raise ValueError("Modality is None. Trainer not applicable for this case.")
    return datasets

def dataset_select(datasets_dict, data_type: str = ""):
    """
    Concatenate list of AnnDatas or Fuse list of AnnDatas into one MuData
    """
    datasets = datasets_dict

    if data_type == "concatenate":  # Process input object for PCA and MultiVI
        concatenate = {}
        for dataset_name, dataset_data in datasets.items():
            print(f"\n=== Concatenating dataset: {dataset_name} ===")
            modalities = dataset_data["modalities"]
            list_anndata = dataset_data["data"]
            data_concat = anndata_concatenate(
                list_anndata=list_anndata, list_modality=modalities
            )
            concatenate[dataset_name] = data_concat
        data = concatenate
    elif data_type == "mudata":  # Process input object for MOFA+ and Mowgli
        mudata_input = {}
        for dataset_name, dataset_data in datasets.items():
            print(f"\n=== Fusing dataset as MuData object: {dataset_name} ===")
            modalities = dataset_data["modalities"]
            list_anndata = dataset_data["data"]
            data_fuse = fuse_mudata(
                list_anndata=list_anndata, list_modality=modalities
            )
            mudata_input[dataset_name] = data_fuse
        data = mudata_input
    else:
        raise ValueError("Only accept datatype of concatenate or mudata.")
    return data
