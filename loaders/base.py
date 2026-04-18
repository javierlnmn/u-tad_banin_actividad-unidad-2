from abc import ABC, abstractmethod

from pandas import DataFrame


class DataLoader(ABC):
    @abstractmethod
    def load(self) -> DataFrame:
        pass
