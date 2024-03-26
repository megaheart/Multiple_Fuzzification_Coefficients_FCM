import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from Components.Cores.sSMC_FCM_Core import sSMC_FCM_Core

class sSMC_FCM_Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4, verbose=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y):
        

    def predict(self, X):
        