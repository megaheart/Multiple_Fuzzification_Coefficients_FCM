import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import torch
import math, time
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(path)
print(path)

C = 100
RANGE_COUNT = 1000
# ### Load the data
print("Load the data")
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
print("Labeling")
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

# Optimize the distance function parameters
def pbm_inversed_loss(X:np.ndarray, Y:np.ndarray):
    """
    inversed PBM index, lower the score, better partitioning, uses for measuring the quality of unsupervised clustering

    Parameters
    ----------
    X : np.ndarray
        2D NumPy array of data
    Y : np.ndarray
        1D NumPy array of labels

    Returns
    -------
    Callable[[Tensor], Tensor]
        Pytorch loss funtion
    """
    N, dim = X.shape
    if N != len(Y):
        raise ValueError("Length of pred and X should be same")
    C = len(np.unique(Y))
    # pred = LabelEncoder().fit_transform(pred)
    
    # Calculate centroids and grand centroid
    centroids = np.zeros((C, dim))
    centroids_count = np.zeros(C)
    grand_centroid = np.mean(X, axis=0)
    for i in range(N):
        centroids[int(Y[i])] += X[i]
        centroids_count[int(Y[i])] += 1
    for i in range(C):
        centroids[i] /= centroids_count[i]
    
    # Calculate E1 - the sum of distances between the elements and the grand centroid of the data
    d_E1 = (grand_centroid - X) ** 2

    # Calculate EK - the sum of within-group distances
    d_EK = np.zeros((N, dim))
    for i in range(N):
        d_EK[i] = (X[i] - centroids[int(Y[i])]) ** 2

    # Calculate DK - the maximum distance between group centroids
    d_DK = np.zeros((int(C * (C - 1) / 2), dim))
    tmp_idx = 0 
    for i in range(C):
        for j in range(i+1, C):
            d_DK[tmp_idx] = (centroids[i] - centroids[j]) ** 2
            tmp_idx += 1
    d_E1 = torch.from_numpy(d_E1).float()
    d_EK = torch.from_numpy(d_EK).float()
    d_DK = torch.from_numpy(d_DK).float()
    d_E1.requires_grad_(False)
    d_EK.requires_grad_(False)
    d_DK.requires_grad_(False)

    # Calculate inversed PBM index
    def inversed_pbm_index(distance_params):
        squared_distance_params = distance_params ** 2
        E1 = torch.sum(torch.sqrt(d_E1.matmul(squared_distance_params)))
        EK = torch.sum(torch.sqrt(d_EK.matmul(squared_distance_params)))
        DK = torch.sum(torch.sqrt(d_DK.matmul(squared_distance_params)))
        pbm_index = EK * C / (E1 * DK)
        return pbm_index * pbm_index + (torch.sum(squared_distance_params) - dim) ** 2
        
    return inversed_pbm_index

print("Prepare variables")
epochs = 1000
loss_fn = pbm_inversed_loss(train_X, Y[:train_size])
distance_params = torch.full((9, 1), 1.0).float()
distance_params.requires_grad_(True)
print(distance_params)
learning_rate = 3e-2
optimizer = torch.optim.Adam([distance_params], lr=learning_rate, betas=(0.5, 0.55))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

print("Start learning params")

for epoch in range(epochs):
    loss = loss_fn(distance_params)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 100 == 99:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: {loss.item()}, lr={lr}")
        with torch.no_grad():
            params = torch.flatten(distance_params)
            print(params)

    # lr = optimizer.param_groups[0]['lr']
    # print(f"Epoch {epoch}: {loss.item()}, lr={lr}")
    # with torch.no_grad():
    #     params = torch.flatten(distance_params)
    #     print(params)
