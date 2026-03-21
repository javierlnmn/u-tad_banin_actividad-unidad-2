from data_loaders.base import DataLoader

import pandas as pd


class CSVLoader(DataLoader):
    def __init__(self, file_path: str, chunksize: int = None):
        self.file_path = file_path
        self.chunksize = chunksize

    def load(self):
        return pd.read_csv(
            self.file_path,
            chunksize=self.chunksize,
            engine="python",
            on_bad_lines="skip",
        )
