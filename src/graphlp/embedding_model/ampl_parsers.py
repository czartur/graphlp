__all__ = ["numpy_to_ampl", "ampl_to_numpy"]

from typing import Tuple, Dict
from itertools import product
import numpy as np


def numpy_to_ampl(matrix: np.ndarray) -> Dict:
    """Expects only 2D arrays as input."""
    shape = matrix.shape
    ampl_matrix = {
        (i+1, j+1): matrix[i][j]
        for i, j in product(range(shape[0]), range(shape[1]))
    }
    return ampl_matrix


def ampl_to_numpy(ampl_matrix, shape: Tuple[int, int]) -> np.ndarray:
    matrix = np.zeros(shape)
    for i, j in product(range(shape[0]), range(shape[1])):
        matrix[i, j] = ampl_matrix[i+1, j+1].value()
    return matrix
