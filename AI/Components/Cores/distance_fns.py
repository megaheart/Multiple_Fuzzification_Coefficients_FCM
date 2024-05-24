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
    # return np.linalg.norm(x - y)
    if y.ndim == 1 and x.ndim == 1:
        return np.sqrt(np.sum((x - y) ** 2))
    elif x.ndim == 2 and y.ndim == 2:
        if x.shape[0] != 1 and x.shape != y.shape:
            raise ValueError("x should have shape (1, n)")
        return np.sqrt(np.sum((x - y) ** 2, axis=1))
    else:
        raise ValueError("Not support x and v")