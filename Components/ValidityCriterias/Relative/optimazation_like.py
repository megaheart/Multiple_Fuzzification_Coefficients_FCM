import numpy as np
from typing import List, Tuple, Callable
from sklearn.preprocessing import LabelEncoder
import Components.Cores.distance_fns as df

def pbm_index(X:np.ndarray, pred:np.ndarray, distance_fn:Callable[[np.ndarray, np.ndarray], float] = df.euclidean_distance)->float:
    """
    PBM index, greater the score, better partitioning, uses for measuring the quality of unsupervised clustering

    Parameters
    ----------
    X : np.ndarray
        2D NumPy array of data
    pred : np.ndarray
        1D NumPy array of predicted labels
    distance_fn : Callable[[np.ndarray, np.ndarray], float], optional
        Distance function, by default Euclidean distance

    Returns
    -------
    float
        PBM index
    """
    n, dim = X.shape
    if n != len(pred):
        raise ValueError("Length of pred and X should be same")
    C = len(np.unique(pred))
    # pred = LabelEncoder().fit_transform(pred)
    
    # Calculate centroids and grand centroid
    centroids = np.zeros((C, dim))
    centroids_count = np.zeros(C)
    grand_centroid = np.mean(X, axis=0)
    for i in range(X):
        centroids[pred[i]] += X[i]
        centroids_count[pred[i]] += 1
    for i in range(C):
        centroids[i] /= centroids_count[i]
    
    # Calculate E1 - the sum of distances between the elements and the grand centroid of the data
    E1 = 0
    for i in range(n):
        E1 += distance_fn(X[i], grand_centroid)

    # Calculate EK - the sum of within-group distances
    EK = 0
    for i in range(n):
        EK += distance_fn(X[i], centroids[pred[i]])

    # Calculate DK - the maximum distance between group centroids
    DK = 0
    for i in range(C):
        for j in range(i+1, C):
            DK = max(DK, distance_fn(centroids[i], centroids[j]))
    
    # Calculate PBM index
    pbm_index = (E1 * DK) / (EK * C)
    return pbm_index * pbm_index

def sswc_score(X:np.ndarray, pred:np.ndarray, distance_fn:Callable[[np.ndarray, np.ndarray], float] = df.euclidean_distance)->float:
    """
    SSWC Score, greater the score, better partitioning, uses for measuring the quality of unsupervised clustering

    Parameters
    ----------
    X : np.ndarray
        2D NumPy array of data
    pred : np.ndarray
        1D NumPy array of predicted labels
    distance_fn : Callable[[np.ndarray, np.ndarray], float], optional
        Distance function, by default Euclidean distance

    Returns
    -------
    float
        SSWC Score
    """
    n, dim = X.shape
    if n != len(pred):
        raise ValueError("Length of pred and X should be same")
    C = len(np.unique(pred))
    # pred = LabelEncoder().fit_transform(pred)
    
    # Calculate centroids
    centroids = np.zeros((C, dim))
    centroids_count = np.zeros(C)
    for i in range(X):
        centroids[pred[i]] += X[i]
        centroids_count[pred[i]] += 1
    for i in range(C):
        centroids[i] /= centroids_count[i]
    
    # Calculate s(x[i]) -> Calculate Index
    sswc_score = 0
    for i in range(n):
        apj = distance_fn(X[i], centroids[pred[i]])
        bpj = np.inf
        for j in range(C):
            if j != pred[i]:
                bpj = min(bpj, distance_fn(X[i], centroids[j]))
        s_xj = (bpj - apj) / max(apj, bpj)
        sswc_score += s_xj
    return sswc_score / n