__all__ = ["IsometricEmbedding"]

import numpy as np

from graphlp.embedding_model.abstract import EmbeddingModel


class IsometricEmbedding(EmbeddingModel):

    def embed(self, graph: np.ndarray) -> np.ndarray:
        n, _ = graph.shape
        return np.random.rand(n, 3)
