import numpy as np
from typing import List, Tuple, Callable, Union
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from ortools.sat.python import cp_model

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

def maximize_accuracy(class_matrix: np.ndarray) -> np.ndarray:
    """
    Maximize number of correct predictions: sum(class_matrix[i, x[i]] for i in range(N))

    Parameters
    ----------
    class_matrix : np.ndarray
        2D Numpy array (C, C) of the number of points in each cluster
        Element (i, j) is the number of points in cluster i of actual label j

    Returns
    -------
    np.ndarray | None
        1D Numpy array of the optimized mapping from predicted cluster index to actual cluster index
        If problem is feasible, return feasible solution
        If the optimization fails, return None
    """
    model = cp_model.CpModel()
    C = class_matrix.shape[0]
    x = [[model.NewIntVar(0, 1, f'x[{i},{j}]') for j in range(C)] for i in range(C)]

    for i in range(C):
        model.Add(sum(x[i][j] for j in range(C)) == 1)
        model.Add(sum(x[j][i] for j in range(C)) == 1)

    model.Maximize(sum(class_matrix[i, j] * x[i][j] for i in range(C) for j in range(C)))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        result = np.full(C, -1, dtype=int)
        for i in range(C):
            for j in range(C):
                if solver.Value(x[i][j]) == 1:
                    result[i] = j
        return result
    elif status == cp_model.FEASIBLE:
        result = np.full(C, -1, dtype=int)
        for i in range(C):
            for j in range(C):
                if solver.Value(x[i][j]) == 1:
                    result[i] = j
    else:
        return None
    

def sync_output_for_nonsupervised_learning(pred:np.ndarray, actual:np.ndarray) -> Union[np.ndarray, None]:
    """
    Synchronize the output for non-supervised learning

    Parameters
    ----------
    pred : np.ndarray
        1D Numpy array of predicted cluster index
    actual : np.ndarray
        1D Numpy array of actual cluster index

    Returns
    -------
    np.ndarray | None
        1D Numpy array of synchronized predicted cluster index
        If the number of clusters in pred and actual are different or 
        the length of pred and actual are different, return None
    """
    pred = pred.astype(int)
    actual = actual.astype(int)
    N = len(pred)
    C = len(np.unique(actual))
    if len(np.unique(pred)) != C or N != len(actual):
        return None

    classes = np.zeros((C, C)).astype(int)

    for i in range(N):
        classes[pred[i], actual[i]] += 1

    mapping = np.argmax(classes, axis=1).flatten()

    # Check if the mapping is correct, np.unique(mapping) == C
    if len(np.unique(mapping)) != C:
        mapping = maximize_accuracy(classes)

    for i in range(N):
        pred[i] = mapping[pred[i]]

    return pred
