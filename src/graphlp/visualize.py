__all__ = ["visualize_embeddings"]

from typing import Callable, List
import numpy as np
import matplotlib.pyplot as plt


def visualize_embeddings(
    embedding_matrix: np.ndarray,
    words: List[str],
    get_word_index: Callable[[str], int]
) -> None:

    # embedding dim should be 2D or 3D
    embedding_dim = embedding_matrix.shape[1]
    assert 2 <= embedding_dim <= 3

    fig = plt.figure()

    if embedding_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:  # dim == 2
        ax = fig.add_subplot(111)

    for word in enumerate(words):
        emb_axis = [embedding_matrix[get_word_index(
            word), k] for k in range(embedding_dim)]
        ax.text(*emb_axis, word, zorder=1)

    plt.show()
