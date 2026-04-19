from abc import ABC, abstractmethod
from pathlib import Path

from pandas import DataFrame


class DataExporter(ABC):
    def __init__(self, df: DataFrame, output_dir: str | Path):
        self.df = df
        self.output_dir = Path(output_dir)

    @abstractmethod
    def export(self) -> None:
        pass
