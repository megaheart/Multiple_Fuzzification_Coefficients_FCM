import numpy as np
import math
import random
from typing import List, Tuple, Callable
from sklearn.metrics import accuracy_score
# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Components.Cores import clusters_generations as cg
import Components.Cores.distance_fns as df
from Components.Cores.MC_FCM_Core import MC_FCM_Core

class MC_FCM(MC_FCM_Core):
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
        MC_FCM_Core.__init__(self)
        self.distance_fn = distance_fn
        self.lnorm = lnorm

    def solve(self, X:np.ndarray, C:int, ml = 1.1, mu = 4.1, epsilon = 0.0001, max_iter = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the MC-FCM algorithm

        Parameters
        ----------
        X : np.ndarray
            2D Numpy array (N, D) of all input points, N is the number of points, D is the number of features
        C: int
            The number of clusters
        ml : float
            Lower bound of fuzziness coefficient
        mu : float
            Upper bound of fuzziness coefficient
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
        self.C = C
        self.U = np.zeros((N, C))
        self.epsilon = epsilon
        self.m, self.V = self.generate_m_and_first_V(ml, mu)

        for l in range(max_iter):
            self.U = self.update_U()
            self.V_old = self.V
            self.V = self.update_V()
            if self.is_converged():
                break

        return self.U, self.V

