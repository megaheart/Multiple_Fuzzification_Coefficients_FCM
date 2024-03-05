import numpy as np
import sympy as sp
from M2_PrecalculationTable import M2_PrecalculationTable
from typing import List, Tuple, Callable
 
class MC_FCM_Core:
    """
    Provide core functions for sSMC-FCM algorithm

    Attributes
    ----------
    X : np.ndarray
        2D Numpy array (N, D) of all input points, N is the number of points, D is the number of features
    V : np.ndarray
        2D Numpy array (K, D) of all cluster centers, K is the number of clusters, D is the number of features
    V_old : np.ndarray
        2D Numpy array (K, D) of all previous cluster centers, K is the number of clusters, D is the number of features
    U : np.ndarray
        2D Numpy array (N, K) of all membership degrees
        Element (i, j) is the membership degree of point i to cluster j
    m : np.ndarray
        Fuzziness coefficient 1D Numpy Array of non-supervised points
    epsilon : float
        Threshold of zero distance
    distance_fn : Callable[[np.ndarray, np.ndarray], float]
        Distance function between two points
    lnorm : int
        Norm of distance function
    """
    X: np.ndarray
    V: np.ndarray
    V_old: np.ndarray
    U: np.ndarray
    m: np.ndarray
    epsilon: float
    distance_fn: Callable[[np.ndarray, np.ndarray], float]
    lnorm: int

    def update_U(self) -> np.ndarray:
        """
        Update dependent variable U 

        Returns
        -------
        np.ndarray
            2D Numpy array (N, K) of all membership degrees
            Element (i, j) is the membership degree of point i to cluster j
        """
        #initialize variables
        X , V , U , m , epsilon , distance_fn , lnorm = self.X , self.V , self.U , self.m , \
            self.epsilon , self.distance_fn , self.lnorm

        # Get the number of points and clusters
        N, C = U.shape
        a = lnorm/(m - 1)
        # Update dependent variable U for non-supervised points
        for i in range(N):
            d = np.ndarray([distance_fn(X[i], V[j]) for j in range(C)])
            min_index = np.argmin(d)

            if d[min_index] < epsilon:
                U[i, :] = 0
                U[i, min_index] = 1
                continue

            delta = 1 / (d ** a[i])
            U[i] = delta / np.sum(delta)

        return U
    
    def update_V(self) -> np.ndarray:
        """
        Update cluster centers V

        Returns
        -------
        np.ndarray
            2D Numpy array (K, D) of all new cluster centers, K is the number of clusters, D is the number of features
        """
        #initialize variables
        X , V , U , m = self.X , self.V , self.U , self.m

        # Calculate new cluster centers V
        N, C = U.shape
        V = np.zeros((C, X.shape[1]))
        for j in range(C):
            u = U[:, j].flatten()
            u = u ** m
            V[j] = np.dot(u, m) / np.sum(u)
        return V

    def check_convergence(self) -> bool:
        """
        Check if the algorithm has converged

        Returns
        -------
        bool
            True if the algorithm has converged, False otherwise
        """
        #initialize variables
        V , V_old, epsilon, distance_fn = self.V , self.V_old , self.epsilon , self.distance_fn

        # Check if the algorithm has converged
        return distance_fn(V, V_old) < epsilon
    
