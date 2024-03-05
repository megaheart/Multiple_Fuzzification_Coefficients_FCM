import numpy as np
import math
import random
from typing import List, Tuple, Callable
from sklearn.metrics import accuracy_score
# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Components.Cores import clusters_generations as cg
import Components.Cores.distance_fns as df
from Components.Cores.sSMC_FCM_Core import sSMC_FCM_Core
from Components.Cores.M2_PrecalculationTable import M2_PrecalculationTable

class sSMC_FCM(sSMC_FCM_Core):
    """
    Algorithm Implimentation of sSMC-FCM algorithm
    """
    def __init__(self, distance_fn:Callable[[np.ndarray, np.ndarray], float] = df.euclidean_distance, lnorm:int = 2):
        """
        Initialize the sSMC-FCM algorithm, please set the distance function and norm of distance function before solving

        Parameters
        ----------
        distance_fn : Callable[[np.ndarray, np.ndarray], float], optional
            Distance function between two points, by default None is Euclidean distance
        lnorm : int
            Norm of distance function, by default 2
        """
        self.distance_fn = distance_fn
        self.lnorm = lnorm

    def solve(self, X:np.ndarray, Y:np.ndarray, C:int, m = 2, alpha = 0.6, epsilon = 0.0001, max_iter = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the sSMC-FCM algorithm

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array (N, D) of all input points, N is the number of points, D is the number of features
        Y : np.ndarray
            1D Numpy array (N) of the cluster index of all input points
            Unless the point is non-supervised, the value is NaN
        C : int
            The number of clusters
        m : float
            Fuzziness coefficient of non-supervised points
        alpha : float
            The expected membership degree value for supervised points
        epsilon : float
            Threshold of zero distance
        max_iter : int
            Maximum number of iterations
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            2D Numpy array (N, C) of all membership degrees
            2D Numpy array (C, D) of all cluster centers
        """
        #initialize variables; initialize U, V, m2
        N = X.shape[0]
        self.X = X
        self.Y = Y
        self.U = np.zeros((N, C))
        self.V = cg.sSMC_FCM_kmean_plus_plus(X, Y, C, self.distance_fn, self.lnorm)
        self.m = m
        self.epsilon = epsilon
        self.U = self.update_U_non_supervision()
        self.m2 = self.calculate_m2(alpha)

        for l in range(max_iter):
            self.U = self.update_U()
            self.V_old = self.V
            self.V = self.update_V()
            if self.is_converged():
                break

        return self.U, self.V

