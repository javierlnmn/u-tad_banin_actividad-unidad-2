from __future__ import annotations

from loaders import CSVLoader, KaggleLoader
from loaders.base import DataLoader


def build_data_loader(
    loader_name: str,
    csv_path: str | None = None,
    kaggle_dataset: str | None = None,
    kaggle_file: str | None = None,
) -> DataLoader:
    if loader_name == "csv":
        return CSVLoader(file_path=csv_path)
    if loader_name == "kaggle":
        return KaggleLoader(dataset_name=kaggle_dataset, file_path=kaggle_file)
    if loader_name == "json":
        raise NotImplementedError("El loader 'json' no está implementado.")
    raise ValueError(f"Loader desconocido: {loader_name}")
