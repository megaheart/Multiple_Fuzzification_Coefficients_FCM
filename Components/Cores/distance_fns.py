import numpy as np

def euclidean_distance(x:np.ndarray, y:np.ndarray) -> float:
    """
    Euclidean distance function

    Parameters
    ----------
    x : np.ndarray
        1D Numpy array of point x
    y : np.ndarray
        1D Numpy array of point y

    Returns
    -------
    float
        The Euclidean distance between x and y
    """
    return np.linalg.norm(x - y)