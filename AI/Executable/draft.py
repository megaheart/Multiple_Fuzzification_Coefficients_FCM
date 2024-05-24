import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

V = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

def distance_fn(x, v):
    if v.ndim == 1 and x.ndim == 1:
        return np.sqrt(np.sum((x - v) ** 2))
    elif x.ndim == 2 and v.ndim == 2:
        if x.shape[0] != 1:
            raise ValueError("x should have shape (1, n)")
        return np.sqrt(np.sum((x - v) ** 2, axis=1))
    else:
        raise ValueError("Not support x and v")

# d = np.array([distance_fn(X[2], V[j]) for j in range(5)])

# d = distance_fn(X[2, None], V)

# np.sqrt(np.sum((V - X[2, None]) ** 2, axis=1))
    
import numpy as np
from sklearn.preprocessing import MinMaxScaler

a = np.array([1, 2, 3, 4, 5, 6])
a_min = np.min(a)
a_max = np.max(a)
a = (a - a_min) / (a_max - a_min) * 999

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

V = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

np.linalg.norm(V - X[2, None])


elements_Qi_min = 2
elements_Qi_max = 7

elements_ranges_Qi = np.linspace(0, 1, 12) * (elements_Qi_max - elements_Qi_min) + elements_Qi_min
