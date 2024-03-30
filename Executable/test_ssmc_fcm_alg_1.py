
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import math, time
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(path)
print(path)
sys.path.append(path)

from Components.Algorithms.sSMC_FCM import sSMC_FCM
import Components.Cores.clusters_generations as cg

if __name__ == '__main__':
    C = 100
    RANGE_COUNT = 1000
    # ### Load the data
    data_path = path + "/Data/battery_cycles.csv"
    dataframe = pd.read_csv(data_path)

    # Split train and test
    test_battery_orders = [82, 16, 4, 96, 36, 32, 29, 18, 14]
    print(test_battery_orders)

    train_battery_orders = [i for i in range(1, 125) if i not in test_battery_orders]

    train_X = dataframe[dataframe['battery_order'].isin(train_battery_orders)]\
        [["c1a_I_dt", "c1a_avg_T", "c1a_avg_I", "c1_max_I","c2_max_I", "c1_max_T", "c1_min_T", "c2_max_T", "c2_min_T"]]\
        .to_numpy()
    train_Qi = dataframe[dataframe['battery_order'].isin(train_battery_orders)][["Qi"]]\
        .to_numpy()
    test_X = dataframe[dataframe['battery_order'].isin(test_battery_orders)]\
        [["c1a_I_dt", "c1a_avg_T", "c1a_avg_I", "c1_max_I","c2_max_I", "c1_max_T", "c1_min_T", "c2_max_T", "c2_min_T"]]\
        .to_numpy()
    test_Qi = dataframe[dataframe['battery_order'].isin(test_battery_orders)][["Qi"]]\
        .to_numpy()

    battery_cycles_count = [len(dataframe[dataframe['battery_order'] == i]) for i in range(1, 125)]
    print(battery_cycles_count)
    print(np.sum(battery_cycles_count))
    
    train_size, test_size = len(train_X), len(test_X)

    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    X = np.concatenate((train_X, test_X), axis=0)
    N = len(X)
    Y = np.full(N, np.nan)
    elements_Qi = np.concatenate((train_Qi, test_Qi), axis=0)

    # Quantiles the Qi
    # region Not used
    # train_Qi_min = np.min(train_Qi)
    # train_Qi_max = np.max(train_Qi)
    # train_Qi_transform = np.round((train_Qi - train_Qi_min) / (train_Qi_max - train_Qi_min) * (RANGE_COUNT - 1))

    # train_Qi_ranges = [[] for _ in range(RANGE_COUNT)]
    # # train_Qi_ranges_Qi = np.linspace(0, 1, RANGE_COUNT) * (train_Qi_max - train_Qi_min) + train_Qi_min

    # for i in range(train_size):
    #     idx = int(train_Qi_transform[i])
    #     train_Qi_ranges[idx].append(i)

    # range_count_tmp = 0
    # cluster_idx = 0
    # clusters_Qi = np.zeros(C)
    # clusters_size = np.zeros(C)
    # for i in range(RANGE_COUNT):
    #     _range = train_Qi_ranges[i]
    #     for x_idx in _range:
    #         Y[x_idx] = cluster_idx
    #         clusters_Qi[cluster_idx] += train_Qi[x_idx]
    #     clusters_size[cluster_idx] += len(_range)  
    #     range_count_tmp += 1
    #     if range_count_tmp % 10 == 0:
    #         cluster_idx += 1
    
    # clusters_Qi = clusters_Qi / clusters_size
    # print("Any cluster don't have any elements: ", np.any(clusters_size == 0))

    # for i in range(C):
    #     if np.any(Y == i):
    #         continue
    #     else:
    #         print(f"Cluster {i} don't have any elements")
    # endregion
    i_Qi = np.argsort(train_Qi, axis=0)
    clusters_Qi = np.zeros(C)

    idx_start = 0
    avg_size = len(train_Qi) // C
    for i in range(C):
        idx_end = i * avg_size + avg_size
        if i == C - 1:
            idx_end = len(train_Qi)
        Y[i_Qi[idx_start:idx_end]] = i
        clusters_Qi[i] = np.mean(train_Qi[i_Qi[idx_start:idx_end]])
        idx_start = idx_end

    for i in range(C):
        if np.any(Y == i):
            continue
        else:
            print(f"Cluster {i} don't have any elements")
    
    # ### Run the sSMC FCM algorithm
    alg = sSMC_FCM()

    # initialize variables; initialize U, V, m2
    print("Initializing variables")
    max_iter = 20
    alpha = 0.5
    alg.X = X
    alg.Y = Y
    alg.U = np.zeros((N, C))
    alg.m = 2
    alg.epsilon = 0.0001
    total_count_time = time.time()
    alg.V = cg.sSMC_FCM_kmean_plus_plus(X, Y, C, alg.distance_fn, alg.lnorm)
    total_count_time = time.time() - total_count_time
    print("Initialized cluster centers in ", total_count_time, " s")

    total_count_time = time.time()
    alg.U = alg.update_U_non_supervision()
    total_count_time = time.time() - total_count_time
    print("Update U non_supervision in ", total_count_time, " s")

    total_count_time = time.time()
    alg.m2 = alg.calculate_m2(alpha)
    total_count_time = time.time() - total_count_time
    print("Initialized m2: ", alg.m2, " in ", total_count_time, " s")

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

    mse_loss = 0
    for i in range(N):
        cluster_index = pred_y[i]
        mse_loss += (clusters_Qi[cluster_index] - elements_Qi[i]) ** 2
    
    print("loss", mse_loss/N)