__all__ = ["UIE"]

import numpy as np

from graphlp.embedding_model.abstract import EmbeddingModel


class UIE(EmbeddingModel):

    def embed(self, graph: np.ndarray) -> np.ndarray:
        n, _ = graph.shape
        return np.random.rand(n, 3)
