__all__ = ["IsometricEmbedding"]

import numpy as np
from sklearn.decomposition import PCA

from graphlp.embedding_model.abstract import EmbeddingModel


def convert_zeros_to_infinity(matrix):
    return np.where(
        (matrix == 0) & ~np.eye(matrix.shape[0], dtype=bool),
        2*matrix.max(),
        matrix
    )


def floyd_warshall(adj_matrix):
    n = adj_matrix.shape[0]
    dist = adj_matrix.copy()
    dist = convert_zeros_to_infinity(dist)
    np.fill_diagonal(dist, 0)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist


def distance_matrix_to_pca(distance_matrix, k):
    n = distance_matrix.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    center_matrix = -0.5 * H.dot(distance_matrix ** 2).dot(H)

    pca = PCA(n_components=k)
    reduced_data = pca.fit_transform(center_matrix)
    return reduced_data


class IsometricEmbedding(EmbeddingModel):
    def __init__(self, kdim: int = 3):
        self._kdim = kdim

    def embed(self, graph: np.ndarray) -> np.ndarray:
        distance_matrix = floyd_warshall(graph)
        pca_matrix = distance_matrix_to_pca(distance_matrix, self._kdim)
        return pca_matrix
