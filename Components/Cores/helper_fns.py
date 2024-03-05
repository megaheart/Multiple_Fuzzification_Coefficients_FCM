import numpy as np

def U_to_cluster_idx(U:np.ndarray) -> np.ndarray:
    """
    Convert membership degrees to cluster index

    Parameters
    ----------
    U : np.ndarray
        2D Numpy array (N, C) of all membership degrees
        Element (i, j) is the membership degree of point i to cluster j

    Returns
    -------
    np.ndarray
        1D Numpy array (N) of the cluster index of all input points
    """
    return np.argmax(U, axis=1)