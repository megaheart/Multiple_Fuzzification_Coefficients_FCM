import json
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import math, time, random
import os
import sys
from Components.Algorithms.sSMC_FCM import sSMC_FCM
import Components.Cores.clusters_generations as cg

def start_predict_capacity_response(connectionId:str):
    res = {
        "isSuccessful": True,
        "connectionId": connectionId,
        "type": "PredictingQi",
        "message": "Start predicting capacity",
        "value": [] # list of double values
    }
    res = json.dumps(res)
    return res
def start_predict_remain_life_response(connectionId:str):
    res = {
        "isSuccessful": True,
        "connectionId": connectionId,
        "type": "PredictingRemainCycle",
        "message": "Start predicting remain life",
        "value": [] # list of double values
    }
    res = json.dumps(res)
    return res
def finish_predict_response(connectionId:str, value: list[float]):
    res = {
        "isSuccessful": True,
        "connectionId": connectionId,
        "type": "ResultAndEvalution",
        "message": "Finish predicting",
        "value": value # list of double values
    }
    res = json.dumps(res)
    return res

def predict_capacity(supervisedBatteryOrders, predictingState, predictingBatteryOrder, predictingCycleOrder):
    C = 10
    
    # ### Load the data
    data_path = "./Data/battery_cycles.csv"
    dataframe = pd.read_csv(data_path)

    terribled_battery_orders = [15, 51, 117, 118, 119, 120]

    train_battery_orders = [i for i in supervisedBatteryOrders if i not in terribled_battery_orders]

    train_X = dataframe[dataframe['battery_order'].isin(train_battery_orders)]\
        [["c1a_I_dt", "c1a_avg_T", "c1a_avg_I", "c1_max_I","c2_max_I", "c1_max_T", "c1_min_T", "c2_max_T", "c2_min_T"]]\
        .to_numpy()
    train_Qi = dataframe[dataframe['battery_order'].isin(train_battery_orders)][["Qi"]]\
        .to_numpy()
    # test_Qi = dataframe[(dataframe['battery_order'] == predictingBatteryOrder) & (dataframe['cycle_order'] == predictingCycleOrder)][["Qi"]]\
    #     .to_numpy().item()

    battery_cycles_count = [len(dataframe[dataframe['battery_order'] == i]) for i in range(1, 125)]
    # print(battery_cycles_count)
    # print(np.sum(battery_cycles_count))

    train_size = len(train_X)

    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X)
    Y = np.full(train_size, np.nan)

    # Quantiles the Qi
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

    # Distance function weights
    alpha_fea = np.array([3.0016, -0.0518,  0.0628,  0.0500, -0.0543, -0.0306,  0.0556, -0.0545, 0.0543])
    alpha_fea = alpha_fea ** 2
    alpha_fea = alpha_fea / np.sum(alpha_fea) * len(alpha_fea)
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
    
    # sSMC-FCM
    _X = [predictingState]
    _X = scaler.transform(_X)
    _Y = [np.nan]
    _X = np.concatenate((train_X, _X), axis=0)
    _Y = np.concatenate((Y[:train_size], _Y), axis=0)
    _N = len(_Y)
    print(len(_X), len(_Y), _N)
    alg = sSMC_FCM(distance_fn=distance_fn)

    # initialize variables; initialize U, V, m2
    print("Initializing variables")
    max_iter = 80
    alpha = 0.5
    alg.X = _X
    alg.Y = _Y
    alg.U = np.zeros((_N, C))
    alg.m = 2
    alg.epsilon = 0.0001
    alg.V = cg.sSMC_FCM_kmean_plus_plus(alg.X, alg.Y, C, alg.distance_fn, alg.lnorm)
    alg.U = alg.update_U_non_supervision()
    alg.m2 = alg.calculate_m2(alpha)
    print("m2 = ", alg.m2)
    # alg.m2 = min(alg.m2, 8)

    for l in range(max_iter):
        alg.U = alg.update_U()
        alg.V_old = alg.V
        alg.V = alg.update_V()
        if alg.is_converged():
            break
        if l % 10 == 9:
            print(f"Iteration {l + 1}/{max_iter} - Different score = {distance_fn(alg.V, alg.V_old)}")

    pred_y = np.argmax(alg.U, axis=1)
    pred_Qi = np.array([clusters_Qi[pred_y[i]] for i in range(train_size, _N)])

    return pred_Qi.flatten()[0]


def predict_remain_life(supervisedBatteryOrders, predictingState, predictingBatteryOrder, predictingCycleOrder, capacity):
    # ### Load the data
    data_path = "./Data/battery_cycles.csv"
    dataframe = pd.read_csv(data_path)
    
    remain_cycles = dataframe[["cycle_order"]].to_numpy()

    begin_idx = 0

    for i in range(1, 125):
        len_cycles = len(dataframe[dataframe['battery_order'] == i])
        end_idx = begin_idx + len_cycles
        remain_cycles[begin_idx:end_idx] = np.arange(len_cycles, 0, -1).reshape(-1, 1)
        begin_idx = end_idx

    dataframe["remain_life"] = remain_cycles

    C = 7

    # Split train and test
    test_battery_orders = [96, 28, 29]
    print(test_battery_orders)

    terribled_battery_orders = [15, 51, 117, 118, 119, 120]

    battery_cycles_count = [len(dataframe[dataframe['battery_order'] == i]) for i in range(1, 125)]
    print(battery_cycles_count)
    print(np.sum(battery_cycles_count))

    train_battery_orders = [i for i in supervisedBatteryOrders if i not in terribled_battery_orders                                                                                                  and battery_cycles_count[i - 1] < 1100]
    print('train_battery_orders: ', train_battery_orders)

    train_X = dataframe[dataframe['battery_order'].isin(train_battery_orders)]\
        [["c1a_I_dt", "c1a_avg_T", "c1a_avg_I", "c1_max_I","c2_max_I", "c1_max_T", "c1_min_T", "c2_max_T", "c2_min_T", "Qi"]]\
        .to_numpy()
    train_t = dataframe[dataframe['battery_order'].isin(train_battery_orders)][["remain_life"]]\
    .to_numpy()

    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X)

    train_size = len(train_X)

    # Quantiles the Qi
    Y = np.full(train_size, np.nan)
    i_t = np.argsort(train_t, axis=0)
    clusters_t = np.zeros(C)

    idx_start = 0
    avg_size = len(train_t) // C
    for i in range(C):
        idx_end = i * avg_size + avg_size
        if i == C - 1:
            idx_end = len(train_t)
        Y[i_t[idx_start:idx_end]] = i
        clusters_t[i] = np.mean(train_t[i_t[idx_start:idx_end]])
        idx_start = idx_end

    for i in range(C):
        if np.any(Y == i):
            continue
        else:
            print(f"Cluster {i} don't have any elements")

    # Distance function weights
    alpha_fea = np.array([6.9586, -2.0383, -2.5299, -1.6623, -7.1471,  0.0532, -0.8207,  0.0669, -0.9456,  5.8426])
    alpha_fea = alpha_fea ** 2
    alpha_fea = alpha_fea / np.sum(alpha_fea) * len(alpha_fea)
    alpha_fea_T = alpha_fea.T
    (np.sum(alpha_fea), alpha_fea, np.round(alpha_fea, 4))

    def distance_fn(x, y):
        if y.ndim == 1 and x.ndim == 1:
            return np.sqrt(np.sum((x - y) ** 2 * alpha_fea))
        elif x.ndim == 2 and y.ndim == 2:
            if x.shape[0] != 1 and x.shape != y.shape:
                raise ValueError("x should have shape (1, n)")
            return np.sqrt(np.dot((x - y) ** 2, alpha_fea_T))
        else:
            raise ValueError("Not support x and v")
        
    # sSMC-FCM
    predictingState = list(predictingState)
    predictingState.append(capacity)
    _X = [predictingState]
    _X = scaler.transform(_X)
    _Y = [np.nan]
    _X = np.concatenate((train_X, _X), axis=0)
    _Y = np.concatenate((Y[:train_size], _Y), axis=0)
    _N = len(_Y)
    print(len(_X), len(_Y), _N)
    alg = sSMC_FCM(distance_fn=distance_fn)
    
    # initialize variables; initialize U, V, m2
    print("Initializing variables")
    max_iter = 10
    alpha = 0.5
    alg.X = _X
    alg.Y = _Y
    alg.U = np.zeros((_N, C))
    alg.m = 2
    alg.epsilon = 0.0001
    alg.V = cg.sSMC_FCM_kmean_plus_plus(alg.X, alg.Y, C, alg.distance_fn, alg.lnorm)
    alg.U = alg.update_U_non_supervision()
    alg.m2 = alg.calculate_m2(alpha)
    print("m2 = ", alg.m2)
    # alg.m2 = min(alg.m2, 8)

    for l in range(max_iter):
        alg.U = alg.update_U()
        alg.V_old = alg.V
        alg.V = alg.update_V()
        if alg.is_converged():
            break
        if l % 10 == 9:
            print(f"Iteration {l + 1}/{max_iter} - Different score = {distance_fn(alg.V, alg.V_old)}")

    pred_y = np.argmax(alg.U, axis=1)
    pred_t = np.array([clusters_t[pred_y[i]] for i in range(train_size, _N)])

    return pred_t.flatten()[0]