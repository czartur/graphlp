from typing import Tuple, Dict
from itertools import product
import numpy as np

# only 2D np.ndarra
def numpy_to_ampl(matrix: np.ndarray) -> Dict:
    shape = matrix.shape 
    ampl_matrix = {
        (i+1, j+1): matrix[i][j]
        for i,j in product(range(shape[0]), range(shape[1]))
        if matrix[i][j] != 0
    }
    return ampl_matrix
    
def ampl_to_numpy(ampl_matrix, shape: Tuple[int,int]) -> np.ndarray:
    matrix = np.zeros(shape)
    for i,j in product(range(shape[0]), range(shape[1])):
            matrix[i,j] = ampl_matrix[i+1,j+1].value()
    return matrix

