__all__ = ["EmbeddingModel"]

from abc import ABC, abstractmethod
import numpy as np


class EmbeddingModel(ABC):

    @abstractmethod
    def embed(self, graph: np.ndarray) -> np.ndarray:
        ...
