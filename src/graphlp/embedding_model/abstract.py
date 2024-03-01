__all__ = ["EmbeddingModel"]

from abc import ABC, abstractmethod
import numpy as np
import networkx as nx


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, graph: nx.Graph, **kwargs) -> np.ndarray:
        ...
