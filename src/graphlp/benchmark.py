__all__ = ["create_error_array", "largest_distance_error"]
import numpy as np

def create_error_array(
    embedding: np.ndarray,
    distance_matrix: np.ndarray,
) -> np.ndarray:  
    n = len(embedding)
    error = np.array([
                abs(np.linalg.norm(embedding[i] - embedding[j]) - distance_matrix[i,j])
                for i in range(n) for j in range(i+1, n)
            ])
    return error

def mean_distance_error(
    embedding: np.ndarray,
    distance_matrix: np.ndarray,
) -> float:
    return create_error_array(embedding, distance_matrix).mean()

def largest_distance_error(
    embedding: np.ndarray,
    distance_matrix: np.ndarray,
) -> float:
    return create_error_array(embedding, distance_matrix).max()

def print_error_summary(
    embedding: np.ndarray,
    distance_matrix: np.ndarray, 
) -> None:
    print(f"mean distance error =", mean_distance_error(embedding, distance_matrix)) 
    print(f"largest distance error =", largest_distance_error(embedding, distance_matrix)) 


