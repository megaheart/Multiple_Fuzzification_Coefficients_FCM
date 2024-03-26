import numpy as np
from typing import List, Tuple, Callable, Union
from queue import PriorityQueue
from multiprocessing.pool import Pool
# import multiprocessing
 
class MC_FCM_Core:
    """
    Provide core functions for sSMC-FCM algorithm

    Attributes
    ----------
    X : np.ndarray
        2D Numpy array (N, D) of all input points, N is the number of points, D is the number of features
    C : int
        The number of clusters
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
    C: int
    V: np.ndarray
    V_old: np.ndarray
    U: np.ndarray
    m: np.ndarray
    epsilon: float
    distance_fn: Callable[[np.ndarray, np.ndarray], float]
    lnorm: int
    pool:Pool

    def __init__(self):
        # if pool is None:
        #     cpu_count = multiprocessing.cpu_count() // 2
        #     self.pool = Pool(cpu_count)
        # else:
        #     self.pool = pool
        pass

    def update_U(self) -> np.ndarray:
        """
        Update dependent variable U

        Time Complexity: O(N*C*D) 

        Returns
        -------
        np.ndarray
            2D Numpy array (N, K) of all membership degrees
            Element (i, j) is the membership degree of point i to cluster j
        """
        # Initialize variables
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

            delta = 1 / (d ** a[i])
            U[i] = delta / np.sum(delta)

        return U 

    def update_V(self) -> np.ndarray:
        """
        Update cluster centers V
        
        Time Complexity: O(N*C*D) 

        Returns
        -------
        np.ndarray
            2D Numpy array (K, D) of all new cluster centers, K is the number of clusters, D is the number of features
        """
        # initialize variables
        X , V , U , m = self.X , self.V , self.U , self.m

        # Calculate new cluster centers V
        N, C = U.shape
        V = np.zeros((C, X.shape[1]))
        for j in range(C):
            u = U[:, j].flatten()
            u = u ** m
            V[j] = np.dot(u, X) / np.sum(u)
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
    
    def generate_m_and_get_nearest_neibor_list(self, ml = 1.1, mu = 4.1, nn: Union[int, None] = None) -> Tuple[np.ndarray, List[PriorityQueue[Tuple[int, float]]], List[int]]:
        """
        Generate fuzziness coefficient m for non-supervised points,
        and get the nearest neighbor list for each point (size = nn)

        Time Complexity: O(N^2*D + N^2*log(nn))

        Parameters
        ----------
        ml : float
            Lower bound of fuzziness coefficient m
        mu : float
            Upper bound of fuzziness coefficient m
        nn : int
            The number of nearest neighbors for each point. If None, nn = N / C

        Returns
        -------
        [0]: np.ndarray
            Fuzziness coefficient 1D Numpy Array
        [1]: List[PriorityQueue]
            List of priority queues, each queue contains the nearest neighbors of a point
        [2]: List[int]
            Element-density-descending index array
        """
        # initialize variables
        X, C, distance_fn = self.X, self.C, self.distance_fn
        # Get the number of points and dimension
        n = len(X)
        nn = n // C if nn is None else nn

        # Calculate density index of each point in X list
        delta2 = np.zeros(n)
        priority_queues = [PriorityQueue(nn) for _ in range(n)]

        def push_to_priority_queues(i, xi, d): # Time Complexity: O(log(nn))
            dd = d
            if priority_queues[i].qsize() < nn:
                priority_queues[i].put((-d, (xi, d)))
            else:
                d_max, tuple_x_d_max = priority_queues[i].get()
                d_max = -d_max
                if(d < d_max):
                    dd -= d_max
                    priority_queues[i].put((-d, (xi, d)))
                else:
                    priority_queues[i].put((-d_max, tuple_x_d_max))
                    dd = 0
            return dd

        # Calculate the distance between two point in X list
        # Time Complexity: O(N^2*D + N^2*log(nn))
        for i in range(n - 1):
            for j in range(i + 1, n):
                d = distance_fn(X[i], X[j])
                delta2[i] += push_to_priority_queues(i, j, d)
                delta2[j] += push_to_priority_queues(j, i, d)

        delta3 = sorted([(delta2[i], i) for i in range(n)], reverse=True) # Time Complexity: N*log(N)

        # Calculate power number of function (alpha)
        delta2_min = delta3[-1][0]
        delta2_max_min = delta3[0][0] - delta2_min
        median = np.median(delta2)
        alpha = np.log(0.5) / np.log((median - delta2_min) / delta2_max_min)

        # Calculate fuzzification coefficients
        m = ml + (mu - ml) * ((delta2 - delta2_min) / delta2_max_min)**alpha # Time Complexity: O(N)

        return m, priority_queues, [delta3[i][1] for i in range(n - 1, -1, -1)]
    
    def generate_m_and_first_V(self, ml = 1.1, mu = 4.1, nn: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fuzziness coefficient m for non-supervised points,
        and generate the first cluster centers V

        Time Complexity: O(N^2*D + N^2*log(nn))

        Parameters
        ----------
        ml : float
            Lower bound of fuzziness coefficient m
        mu : float
            Upper bound of fuzziness coefficient m
        nn : int
            The number of nearest neighbors for each point. If None, nn = N / C

        Returns
        -------
        [0]: np.ndarray
            Fuzziness coefficient 1D Numpy Array
        [1]: np.ndarray
            2D Numpy array (C, D) of all new cluster centers, C is the number of clusters, D is the number of features
        """
        m, priority_queues, density_idxs = self.generate_m_and_get_nearest_neibor_list(ml, mu, nn)
        self.m = m
        # Get the number of clusters and dimension
        C = self.C
        n, dimension = self.X.shape
        X = self.X
        # Generate the first cluster centers V
        xis:List[int] = []
        z_indexs = np.full(n, n)
        max_groups_v = np.empty((n, dimension))
        # xis.clear()
        for xi in density_idxs:
            if len(xis) >= C:
                break
            if z_indexs[xi] != n:
                continue
            xis.append(xi)
            max_group_size = 1
            z_indexs[xi] = xi
            max_groups_v[xi] = np.copy(X[xi])
            if priority_queues[xi].qsize() > 0:
                priority_queues[xi].get()
            while priority_queues[xi].qsize() > 0:
                _, p = priority_queues[xi].get()
                if z_indexs[p[0]] != n:
                    continue
                z_indexs[p[0]] = xi
                max_group_size += 1
                max_groups_v[xi] += X[p[0]]
            max_groups_v[xi] /= max_group_size

        V = max_groups_v[xis]
        return m, V
            
    
    def generate_m(self, ml = 1.1, mu = 4.1, nn: int = None) -> np.ndarray:
        """
        Generate fuzziness coefficient m for non-supervised points

        Time Complexity: 

        Parameters
        ----------
        ml : float
            Lower bound of fuzziness coefficient m
        mu : float
            Upper bound of fuzziness coefficient m
        nn : int
            The number of nearest neighbors for each point. If None, nn = N / C

        Returns
        -------
        np.ndarray
            Fuzziness coefficient 1D Numpy Array
        """
        return self.generate_m_and_get_nearest_neibor_list(ml, mu, nn)[0]

    
