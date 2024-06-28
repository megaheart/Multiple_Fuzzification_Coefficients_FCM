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

def quantize(self, x):
    C = 10
    RANGE_COUNT = 1000
    # ### Load the data
    data_path = "./Data/battery_cycles.csv"
    dataframe = pd.read_csv(data_path)

    # Split train and test
    test_battery_orders = [96, 28, 29]
    print(test_battery_orders)

    terribled_battery_orders = [15, 51, 117, 118, 119, 120]

    train_battery_orders = [i for i in range(1, 125) if i not in test_battery_orders and i not in terribled_battery_orders]

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

    return x