import numpy as np
import math
import random
from typing import List, Tuple, Callable
from Components.Cores.distance_fns import euclidean_distance

def propor_select(vals:np.ndarray) -> int:
    """
        Roulette wheel selection

        Parameters
        ----------
        vals: List of probabilities for each element

        Returns
        -------
        int
            The index of the selected element
    """
    # roulette wheel selection
    # on the fly technique
    # vals[] can't be all 0.0s
    n = len(vals)
    vals = vals / np.sum(vals)

    cum_p = 0.0  # cumulative prob
    p = random.random()

    for i in range(n):
        cum_p += vals[i]
        if cum_p > p:
            return i
    return n - 1  # last index

def kmean_plus_plus(X:np.ndarray, C:int, V:np.ndarray = None, cluster_idxs_init:list[int] = None, \
                    distance_fn:Callable[[np.ndarray, np.ndarray], float] = euclidean_distance, lnorm:float = 2) -> np.ndarray:
    """
    KMean++ algorithm

    Time Complexity: O(C^2*N*D)

    Parameters
    ----------
    X : np.ndarray
        2D Numpy array (N, D) of all input points, N is the number of points, D is the number of features
    C : int
        The number of clusters
    cluster_idxs_init : list[int]
        The index of the initial cluster centers, the algorithm will not reinitialize these clusters and
        will only initialize the remaining clusters
    distance_fn : Callable[[np.ndarray, np.ndarray], float]
        Distance function between two points
    lnorm : float
        Norm of distance function

    Returns
    -------
    np.ndarray
        2D Numpy array (C, D) of the cluster centers
    """
    N, dim = X.shape

    if V is None:
        V = np.zeros((C, dim))
    if cluster_idxs_init is None:
        cluster_idxs_init = []

    cluster_idxs_not_init = np.setdiff1d(np.arange(C), cluster_idxs_init).tolist()

    if len(cluster_idxs_not_init) == C:
        idx = random.randint(0, N-1)
        c_idx = cluster_idxs_not_init.pop()
        V[c_idx] = X[idx]
        cluster_idxs_init.append(c_idx)

    while len(cluster_idxs_not_init) > 0:
        cluster_idx = cluster_idxs_not_init.pop()
        d_squareds = np.full(N, math.inf, dtype=float)
        for i in range(N):
            for ki in cluster_idxs_init:
                d_squareds[i] = min(d_squareds[i], distance_fn(X[i], V[ki]) ** lnorm)

        new_cluster_idx = propor_select(d_squareds)
        V[cluster_idx] = X[new_cluster_idx]
        cluster_idxs_init.append(cluster_idx)

    return V

def sSMC_FCM_kmean_plus_plus(X:np.ndarray, Y:np.ndarray, C:int, distance_fn:Callable[[np.ndarray, np.ndarray], float], lnorm:float) -> np.ndarray:
    """
    Custom KMean++ algorithm for sSMC-FCM

    Time Complexity: O(N*D) if all cluster have supervised points, otherwise O(C^2*N*D)

    Parameters
    ----------
    X : np.ndarray
        2D Numpy array (N, D) of all input points, N is the number of points, D is the number of features
    Y : np.ndarray
        1D Numpy array (N) of the cluster index of all input points
        Unless the point is non-supervised, the value is NaN
    C : int
        The number of clusters
    distance_fn : Callable[[np.ndarray, np.ndarray], float]
        Distance function between two points
    lnorm : float
        Norm of distance function
    """
    N, dim = X.shape

    V = np.zeros((C, dim))
    V_count = np.zeros(C, dtype=int)

    for i in range(N):
        if not np.isnan(Y[i]):
            cluster_idx = int(Y[i])
            V_count[cluster_idx] += 1
            V[cluster_idx] += X[i]

    cluster_idxs_not_init = []
    cluster_idxs_init = []
    for k in range(C):
        if V_count[k] == 0:
            cluster_idxs_not_init.append(k)
        else:
            V[k] /= V_count[k]
            cluster_idxs_init.append(k)

    if len(cluster_idxs_not_init) > 0:
        V = kmean_plus_plus(X, C, V, cluster_idxs_init, distance_fn, lnorm)

    return V
