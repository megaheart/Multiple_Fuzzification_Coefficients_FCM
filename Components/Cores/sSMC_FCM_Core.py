import numpy as np
import sympy as sp
import scipy.optimize as opt
from Components.Cores.M2_PrecalculationTable import M2_PrecalculationTable
from typing import List, Tuple, Callable

class sSMC_FCM_Core:
    """
    Provide core functions for sSMC-FCM algorithm

    Attributes
    ----------
    X : np.ndarray
        2D Numpy array (N, D) of all input points, N is the number of points, D is the number of features
    Y : np.ndarray
        1D Numpy array (N) of the cluster index of all input points
        Unless the point is non-supervised, the value is NaN
    V : np.ndarray
        2D Numpy array (C, D) of all cluster centers, C is the number of clusters, D is the number of features
    V_old : np.ndarray
        2D Numpy array (C, D) of all previous cluster centers, C is the number of clusters, D is the number of features
    U : np.ndarray
        2D Numpy array (N, C) of all membership degrees
        Element (i, j) is the membership degree of point i to cluster j
    m : float
        Fuzziness coefficient of non-supervised points
    m2 : float
        Fuzziness coefficient of supervised points
    epsilon : float
        Threshold of zero distance
    distance_fn : Callable[[np.ndarray, np.ndarray], float]
        Distance function between two points
    lnorm : int
        Norm of distance function
    """
    X: np.ndarray
    Y: np.ndarray
    V: np.ndarray
    V_old: np.ndarray
    U: np.ndarray
    m: float
    m2: float
    epsilon: float
    distance_fn: Callable[[np.ndarray, np.ndarray], float]
    lnorm: int
    __m2_table: M2_PrecalculationTable

    def update_U_non_supervision(self) -> np.ndarray:
        """
        Update dependent variable U without supervision (without label Y)

        Time Complexity: O(N*C*D)

        Returns
        -------
        np.ndarray
            2D Numpy array (N, C) of all membership degrees
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
            d = distance_fn(X[i, None], V)
            min_index = np.argmin(d)

            if d[min_index] < epsilon:
                U[i, :] = 0
                U[i, min_index] = 1
                continue
            
            delta = 1 / (d ** a)
            U[i] = delta / np.sum(delta)
        return U

    def update_U(self) -> np.ndarray:
        """
        Update dependent variable U 

        Time complexity: O(N*C*D)

        Returns
        -------
        np.ndarray
            2D Numpy array (N, C) of all membership degrees
            Element (i, j) is the membership degree of point i to cluster j
        """
        #initialize variables
        X , Y , V , U , m , m2 , epsilon , distance_fn , lnorm = self.X , self.Y , self.V , self.U , self.m , \
            self.m2 , self.epsilon , self.distance_fn , self.lnorm

        # Get the number of points and clusters
        N, C = U.shape
        a = lnorm/(m - 1)
        a2 = lnorm/(m2 - 1)
        a3 = (m2 - m) / (m2 - 1)
        rm = m ** (1 / lnorm)
        rm2 = m2 ** (1 / lnorm)
        # Update dependent variable U for non-supervised points
        for i in range(N):
            d = distance_fn(X[i, None], V)
            min_index = np.argmin(d)

            if d[min_index] < epsilon:
                U[i, :] = 0
                U[i, min_index] = 1
                continue

            if np.isnan(Y[i]): # Non-supervised point
                delta = 1 / (d ** a)
                U[i] = delta / np.sum(delta)

            else: # Supervised point
                k = int(Y[i])
                min_d = d[min_index]
                delta = d / min_d
                mu = 1 / ((rm * d) ** a)
                sum_mu_not_k = np.sum(mu) - mu[k]
                eq_right = 1 / ((rm2 * d[k]) ** a2)
                # mu_k_sym = sp.symbols('mu_k')
                # eq = sp.Eq(mu_k_sym / ((mu_k_sym + sum_mu_not_k) ** a3) , eq_right)
                # mu_k = sp.solve(eq, mu_k_sym)
                mu_k = opt.root_scalar(lambda mu_k: mu_k / ((mu_k + sum_mu_not_k) ** a3) - eq_right, \
                                       bracket=[0, 1000], method='brentq')
                # print(eq_right, sum_mu_not_k, a3)
                # print(mu_k)
                # raise Exception("Stop here")
                mu[k] = mu_k.root
                U[i] = mu / np.sum(mu)

        return U
    
    def update_V(self) -> np.ndarray:
        """
        Update cluster centers V

        Time Complexity: O(N*C*D)

        Returns
        -------
        np.ndarray
            2D Numpy array (C, D) of all new cluster centers, C is the number of clusters, D is the number of features
        """
        #initialize variables
        X , Y , V , U , m , m2 = self.X , self.Y , self.V , self.U , self.m , self.m2

        # Calculate new cluster centers V
        N, C = U.shape
        V = np.zeros((C, X.shape[1])) # Time complexity: O(C*D)
        for j in range(C):
            u = U[:, j].flatten() # Time complexity: O(N)
            # m_arr = [m if np.isnan(Y[i]) or (Y[i] != j) else m2 for i in range(N)]
            m_arr = m + (m2 - m) * (Y == j)
            u = u ** m_arr
            V[j] = np.dot(u, X) / np.sum(u) # Time complexity: O(N*D)
        return V

    def is_converged(self) -> bool:
        """
        Check whether if the algorithm has converged

        Returns
        -------
        bool
            True if the algorithm has converged, False otherwise
        """
        #initialize variables
        V , V_old, epsilon, distance_fn = self.V , self.V_old , self.epsilon , self.distance_fn

        # Check if the algorithm has converged
        return np.all(distance_fn(V, V_old) < epsilon)
    
    def calculate_m2(self, alpha = 0.6) -> float:
        """
        Calculate m2 value

        Time Complexity: O(len(Y)) ~ O(N)

        Parameters
        ----------
        alpha : float
            The expected membership degree value for supervised points

        Returns
        -------
        float
            The precalculated m2 value
        """
        #initialize variables
        Y, m, U = self.Y, self.m, self.U
        u = [U[i, int(Y[i])] for i in range(len(Y)) if not np.isnan(Y[i])]
        if len(u) == 0:
            return m    
        u_min = np.min(u)
        return M2_PrecalculationTable.calculate_m2_integer(m, m, alpha, u_min) # Time Complexity: O(1)
        # if self.__m2_table is None:
        #     self.__m2_table = M2_PrecalculationTable(m, alpha)
        # # Get the precalculated m2 value
        # u = [U[i, int(Y[i])] for i in range(len(Y)) if not np.isnan(Y[i])]
        # if len(u) == 0:
        #     return m    
        # u_min = np.min(u)

        # return self.__m2_table[u_min]
    
