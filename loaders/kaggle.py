import kagglehub
from kagglehub import KaggleDatasetAdapter

from loaders.base import DataLoader


class KaggleLoader(DataLoader):
    def __init__(self, dataset_name: str, file_path: str):
        self.dataset_name = dataset_name
        self.file_path = file_path

    def load(self):
        return kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            self.dataset_name,
            self.file_path,
            pandas_kwargs={
                "lineterminator": "\n",
            },
        )
