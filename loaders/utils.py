from __future__ import annotations

from enum import Enum

from loaders import CSVLoader, KaggleLoader, RapidAPITwitterLoader
from loaders.base import DataLoader


class Loaders(Enum):
    CSV = "csv"
    KAGGLE = "kaggle"
    RAPIDAPI = "rapidapi"
    JSON = "json"


def build_data_loader(
    loader_name: Loaders,
    csv_path: str | None = None,
    kaggle_dataset: str | None = None,
    kaggle_file: str | None = None,
    rapidapi_tweet_count: int = 300,
    rapidapi_use_file: bool = False,
) -> DataLoader:
    if loader_name == Loaders.CSV:
        return CSVLoader(file_path=csv_path)

    if loader_name == Loaders.KAGGLE:
        return KaggleLoader(dataset_name=kaggle_dataset, file_path=kaggle_file)

    if loader_name == Loaders.RAPIDAPI:
        return RapidAPITwitterLoader(
            tweet_count=rapidapi_tweet_count, use_file=rapidapi_use_file
        )

    if loader_name == Loaders.JSON:
        raise NotImplementedError("El loader 'json' no está implementado.")

    raise ValueError(f"Loader desconocido: {loader_name}")
