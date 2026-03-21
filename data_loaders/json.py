from data_loaders.base import DataLoader

import pandas as pd


class JSONLoader(DataLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        return pd.read_json(self.file_path)
