import numpy as np
from typing import List, Tuple, Callable
from sklearn.metrics import jaccard_score, rand_score, accuracy_score, davies_bouldin_score, calinski_harabasz_score, silhouette_score

from Components.ValidityCriterias.Relative.optimazation_like import pbm_index, sswc_score
import Components.Cores.distance_fns as df

def evalute(X:np.ndarray, pred:np.ndarray, true:np.ndarray, distance_fn:Callable[[np.ndarray, np.ndarray], float] = df.euclidean_distance)\
    ->Tuple[float, float, float, float, float, float]:
    """
    Evaluate the clustering algorithm using various validity criterias

    Parameters
    ----------
    X : np.ndarray
        2D NumPy array of data
    pred : np.ndarray
        1D NumPy array of predicted labels
    true : np.ndarray
        1D NumPy array of true labels
    distance_fn : Callable[[np.ndarray, np.ndarray], float], optional
        Distance function, by default Euclidean distance

    Returns
    -------
    [0]: float
        Sum of Squared Within-Cluster Distance (SSWC), relative metric, greater the score, better partitioning
    [1]: float
        Davies-Bouldin index, relative metric, lesser the score, better partitioning
    [2]: float
        PBM index, relative metric, greater the score, better partitioning
    [3]: float
        Accuracy, external metric, greater the score, better partitioning
    [4]: float
        RAND index, external metric, greater the score, better partitioning
    [5]: float
        Jaccard Similarity Score, external metric, greater the score, better partitioning
    """
    sswc = sswc_score(X, pred, distance_fn)
    db = davies_bouldin_score(X, pred)
    pbm = pbm_index(X, pred, distance_fn)
    accuracy = accuracy_score(true, pred)
    rand = rand_score(true, pred)
    jaccard = jaccard_score(true, pred)

    return sswc, db, pbm, accuracy, rand, jaccard