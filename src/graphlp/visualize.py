
from typing import Callable, List
import numpy as np


def visualize_embeddings(
    embedding_matrix: np.ndarray,
    words: List[str],
    get_word_index: Callable[[str], int]
) -> None:
    ...

