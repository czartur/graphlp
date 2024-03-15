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

    embeddings = np.array(
        [embedding_matrix[get_word_index(word)] for word in words])
    min_vals = np.min(embeddings, axis=0)
    max_vals = np.max(embeddings, axis=0)
    min_vals = np.where(min_vals > 0, min_vals * 0.5, min_vals * 2)
    max_vals = np.where(max_vals > 0, max_vals * 2, max_vals * 0.5)

    for i, word in enumerate(words):
        ax.text(*embeddings[i, :embedding_dim], word, zorder=1)
        ax.scatter(*embeddings[i, :embedding_dim], s=10, zorder=2)

    ax.set_xlim(min_vals[0], max_vals[0])
    ax.set_ylim(min_vals[1], max_vals[1])
    if embedding_dim == 3:
        ax.set_zlim(min_vals[2], max_vals[2])

    plt.draw()
