import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import torch
import math, time, random
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(path)
print(path)
sys.path.append(path)

from Components.Cores.clusters_generations import sSMC_FCM_kmean_plus_plus

C = 100
RANGE_COUNT = 1000
# ### Load the data
print("Load the data")
data_path = path + "/Data/battery_cycles.csv"
dataframe = pd.read_csv(data_path)

remain_cycles = dataframe[["cycle_order"]].to_numpy()

begin_idx = 0

for i in range(1, 125):
  len_cycles = len(dataframe[dataframe['battery_order'] == i])
  end_idx = begin_idx + len_cycles
  remain_cycles[begin_idx:end_idx] = np.arange(len_cycles, 0, -1).reshape(-1, 1)
  begin_idx = end_idx

dataframe["remain_life"] = remain_cycles

# Split train and test
test_battery_orders = [96, 28, 29]
print(test_battery_orders)
battery_cycles_count = [len(dataframe[dataframe['battery_order'] == i]) for i in range(1, 125)]
terribled_battery_orders = [15, 51, 117, 118, 119, 120]

train_battery_orders = [i for i in range(1, 125) if i not in test_battery_orders and i not in terribled_battery_orders                                                             and battery_cycles_count[i - 1] < 900]

train_X = dataframe[dataframe['battery_order'].isin(train_battery_orders)]\
    [["c1a_I_dt", "c1a_avg_T", "c1a_avg_I", "c1_max_I","c2_max_I", "c1_max_T", "c1_min_T", "c2_max_T", "c2_min_T", "Qi"]]\
    .to_numpy()
train_t = dataframe[dataframe['battery_order'].isin(train_battery_orders)][["remain_life"]]\
    .to_numpy()
test_X = dataframe[dataframe['battery_order'].isin(test_battery_orders)]\
    [["c1a_I_dt", "c1a_avg_T", "c1a_avg_I", "c1_max_I","c2_max_I", "c1_max_T", "c1_min_T", "c2_max_T", "c2_min_T", "Qi"]]\
    .to_numpy()
test_t = dataframe[dataframe['battery_order'].isin(test_battery_orders)][["remain_life"]]\
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
elements_t = np.concatenate((train_t, test_t), axis=0)

# Quantiles the Qi
print("Labeling")
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

# Split validated set
train_idxs, validate_idxs = next(KFold(random_state=38, shuffle=True).split(range(train_size)))

train_X, validate_X = train_X[train_idxs], train_X[validate_idxs]
train_Y, validate_Y = Y[train_idxs], Y[validate_idxs]
train_size, validate_size = len(train_X), len(validate_X)

class BatteryDataset(Dataset):
    def __init__(self, elements, labels, cluster_vectors):
        self.elements = elements
        self.labels = labels
        self.cluster_vectors = cluster_vectors
        self.items = []
        self._generate_triplet_list_with_cluster(elements, labels, cluster_vectors)

        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, item):
        return self.items[item]
    
    def _generate_triplet_list_with_cluster(self, elements, labels, cluster_vectors):
        for i in range(len(elements)):
            anchor_element = elements[i]
            anchor_label = int(labels[i])

            positive_element = cluster_vectors[anchor_label]
            for i in range(len(cluster_vectors)):
                if i != anchor_label:
                    self.items.append((anchor_element, positive_element, cluster_vectors[i]))
    
    def _ramdom_generate_triplet(self, item):
        anchor_label = self.labels[item]

        positive_idx_list = np.where(self.labels == anchor_label)[0]
        positive_idx_list = positive_idx_list[positive_idx_list != item]
        
        positive_item = random.choice(positive_idx_list)
        
        negative_idx_list = np.where(self.labels != anchor_label)[0]
        negative_item = random.choice(negative_idx_list)

        anchor_element = self.elements[item]
        positive_element = self.elements[positive_item]
        negative_element = self.elements[negative_item]
        
        return anchor_element, positive_element, negative_element
        
class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1: torch.Tensor, x2: torch.Tensor):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, \
                    distance_params: torch.Tensor) -> torch.Tensor:
        dim = 10
        # regularization = 1000 * (torch.sum(distance_params ** 2) - dim) ** 2

        distance_params = distance_params / torch.sum(distance_params) * 10

        anchor = anchor * distance_params
        positive = positive * distance_params
        negative = negative * distance_params

        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        # regularization = 1000 * (torch.sum(distance_params ** 2) - dim) ** 2

        return losses.mean()


# Initialize cluster vector
print("Initialize cluster vector")
V = sSMC_FCM_kmean_plus_plus(train_X, train_Y, C, distance_fn=None, lnorm=None)

print("Prepare variables")
epochs = 10
loss_fn = torch.jit.script(TripletLoss())
batch_size = 128
# distance_params = torch.tensor([0.037361976, 0.03889106, 0.0363686 , 0.17359094, 0.01109282,
#        0.0151649 , 0.00379076, 0.0188133 , 0.00453225, 0.32413561]).float()
distance_params = torch.full((10,), 1.0, requires_grad=True)
distance_params.requires_grad_(True)
print(distance_params)
learning_rate = 5e-2
optimizer = torch.optim.Adam([distance_params], lr=learning_rate, betas=(0.5, 0.55))
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

train_dataset = BatteryDataset(validate_X, validate_Y, V)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)

print("Start learning params")

# for anchor_element, positive_element, negative_element in train_dataloader:
#         anchor_element = anchor_element * distance_params
#         positive_element = positive_element * distance_params
#         negative_element = negative_element * distance_params

#         loss = loss_fn(anchor_element, positive_element, negative_element, distance_params)

#         break

# exit()
for epoch in range(epochs):
    loss_value = 0
    for anchor_element, positive_element, negative_element in train_dataloader:
        # anchor_element = anchor_element * distance_params
        # positive_element = positive_element * distance_params
        # negative_element = negative_element * distance_params

        loss = loss_fn(anchor_element, positive_element, negative_element, distance_params)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

    # scheduler.step()

    # if epoch % 100 == 99:
    #     lr = optimizer.param_groups[0]['lr']
    #     print(f"Epoch {epoch}: {loss.item()}, lr={lr}")
    #     with torch.no_grad():
    #         params = torch.flatten(distance_params)
    #         print(params)

    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: {loss_value}, lr={lr}")
    with torch.no_grad():
        params = torch.flatten(distance_params)
        regularization = 1000 * (torch.sum(distance_params ** 2) - 10) ** 2
        print(params, regularization)
