import numpy as np
from typing import List, Tuple

# NOTE: sklearn.metrics has a function for this, please use that instead

def jaccard_rand_score(pred:np.ndarray, true:np.ndarray)->Tuple[float, float]:
    """
    Jaccard Similarity Score and RAND index (<= 1), greater the score, better partitioning

    Parameters
    ----------
    pred : np.ndarray
        1D NumPy array of predicted labels
    true : np.ndarray
        1D NumPy array of true labels

    Returns
    -------
    [0]: float
        Jaccard Similarity Score
    [1]: float
        Rand Index
    """
    a = b = c = d = 0
    N = len(pred)
    for i in range(N):
        for j in range(i + 1, N):
            has_same_class_in_expect = true[i] == true[j]  # belonging to the same class in R - expect
            in_same_cluster_in_result = pred[i] == pred[j]  # belonging to the same cluster in Q - result
            if has_same_class_in_expect:
                if in_same_cluster_in_result:
                    a += 1
                else:
                    b += 1
            else:
                if in_same_cluster_in_result:
                    c += 1
                else:
                    d += 1
    jaccard_index = a / (a + b + c)
    rand_index = (a + d) / (a + b + c + d)
    return jaccard_index, rand_index
    