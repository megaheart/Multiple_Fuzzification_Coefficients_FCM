import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from Components.Cores.sSMC_FCM_Core import sSMC_FCM_Core
from Components.Cores.MC_FCM_Core import MC_FCM_Core
# from Components.Cores.distance_fns import euclidean_distance

def custom_distance_fn(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class Regressor1(BaseEstimator, RegressorMixin):
    def __init__(self, n_clusters=100, max_iter=100, epsilon=1e-4):
        self.fcm = MC_FCM_Core()
        self.fcm.C = n_clusters
        self.fcm.max_iter = max_iter
        self.fcm.epsilon = epsilon
        self.fcm.lnorm = 2 
        self.fcm.distance_fn = custom_distance_fn

    def fit(self, X, y):
        

    def predict(self, X):
        