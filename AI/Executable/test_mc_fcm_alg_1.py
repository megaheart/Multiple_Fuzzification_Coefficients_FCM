# %% [markdown]
# # sSMC FCM Algorithm and Experiment
# ## Importing Libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
import pandas as pd
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
import math, time
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(path)
print(path)
sys.path.append(path)

# "c1a_I_dt", "c1a_avg_T", "c1a_avg_I", "c1_max_I", "c2_max_I", 
# "c1_max_T", "c1_min_T", "c2_max_T", "c2_min_T"
alpha_fea = np.array([0.45, 0.225, 0.225, 0.025, 0.025, 0.0125, 0.0125, 0.0125, 0.0125])
alpha_fea_T = alpha_fea.T

def distance_fn(x, y):
    if y.ndim == 1 and x.ndim == 1:
        return np.sqrt(np.sum((x - y) ** 2 * alpha_fea))
    elif x.ndim == 2 and y.ndim == 2:
        if x.shape[0] != 1 and x.shape != y.shape:
            raise ValueError("x should have shape (1, n)")
        return np.sqrt(np.dot((x - y) ** 2, alpha_fea_T))
    else:
        raise ValueError("Not support x and v")

# %%
from Components.Algorithms.MC_FCM import MC_FCM
import Components.Cores.clusters_generations as cg
import Components.Cores.helper_fns as hepler

if __name__ == '__main__':
    # %% [markdown]
    # ## Test the sSMC FCM algorithm

    # %% [markdown]
    # ### Load the data

    # %%
    data_path = path + "/Data/battery_cycles.csv"
    dataframe = pd.read_csv(data_path)
    elements = dataframe[["c1a_I_dt", "c1a_avg_T", "c1a_avg_I", "c1_max_I", "c2_max_I", "c1_max_T", "c1_min_T", "c2_max_T", "c2_min_T"]].to_numpy()
    elements_Qi = dataframe[["Qi"]].to_numpy()

    # %% [markdown]
    # ### Run the sSMC FCM algorithm

    # %%
    alg = MC_FCM(distance_fn=distance_fn, lnorm=2)
    # prepare data
    X = elements
    C = 100

    # %%
    # initialize variables; initialize U, V, m2
    print("Initializing variables")
    N = X.shape[0]
    alg.X = X
    alg.C = C
    alg.U = np.zeros((N, C))
    alg.epsilon = 0.0001
    max_iter = 1000
    print("Initializing cluster centers and m")
    count_time = time.time()
    # alg.m, alg.V = alg.generate_m_and_first_V(ml = 1.1, mu=4.1)
    alg.m = np.full(N, 2)
    alg.V = kmeans_plusplus(X, C)[0]
    count_time = time.time() - count_time
    print(f"Time consuming: {count_time} s")

    total_count_time = time.time()
    print("Start solving the algorithm")
    for l in range(max_iter):
        count_time = time.time()
        alg.U = alg.update_U()
        U_count_time = time.time() - count_time

        alg.V_old = alg.V

        count_time = time.time()
        alg.V = alg.update_V()
        V_count_time = time.time() - count_time

        print(f"Iteration {l}/{max_iter} completed in {U_count_time + V_count_time} s")
        print(f"U, V calculation time consume: {U_count_time} s, {V_count_time} s")

        if alg.is_converged():
            break
    total_count_time = time.time() - total_count_time
    print(f"Time consuming: {total_count_time} s")

    pred_y = np.argmax(alg.U, axis=1)
    y_clusters = np.zeros(C)
    y_clusters_count = np.zeros(C)
    for i in range(N):
        cluster_index = pred_y[i]
        y_clusters[cluster_index] += elements_Qi[i]
        y_clusters_count[cluster_index] += 1

    y_clusters = y_clusters / y_clusters_count
    
    mse_loss = 0

    for i in range(N):
        cluster_index = pred_y[i]
        mse_loss += (y_clusters[cluster_index] - elements_Qi[i]) ** 2
    
    print("loss", mse_loss)