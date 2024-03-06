# %% [markdown]
# # sSMC FCM Algorithm and Experiment
# ## Importing Libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
import math
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(path)
print(path)
sys.path.append(path)

# %%
from Components.Algorithms.sSMC_FCM import sSMC_FCM
import Components.Cores.clusters_generations as cg

# %% [markdown]
# ## Test the sSMC FCM algorithm

# %% [markdown]
# ### Load the data

# %%
data_path = path + "/Data/iris.data"
dataframe = pd.read_csv(data_path)
elements = dataframe.iloc[:, :-1].values
label = dataframe.iloc[:, -1].values
label = pre.LabelEncoder().fit_transform(label)
C = len(np.unique(label))

# %% [markdown]
# ### Run the sSMC FCM algorithm

# %%
alg = sSMC_FCM()
# prepare data
train_X, test_X, train_y, test_y = train_test_split(
    elements, label, test_size=0.2, random_state=42)
train_size, test_size = len(train_X), len(test_X)
X = np.concatenate((train_X, test_X), axis=0)
Y = np.concatenate((train_y, np.array([math.nan]*test_size)), axis=0)

print("Number of unique labels in training, test: ",
      len(np.unique(train_y)), len(np.unique(test_y)))

# %%
# initialize variables; initialize U, V, m2
print("Initializing variables")
max_iter = 200
alpha = 0.5
N = elements.shape[0]
alg.X = X
alg.Y = Y
alg.U = np.zeros((N, C))
alg.m = 2
alg.epsilon = 0.0001
print("Initializing cluster centers")
alg.V = cg.sSMC_FCM_kmean_plus_plus(elements, Y, C, alg.distance_fn, alg.lnorm)
print("Initializing membership degrees")
alg.U = alg.update_U_non_supervision()
alg.m2 = alg.calculate_m2(alpha)
print("Initialized m2: ", alg.m2)

for l in range(max_iter):
    alg.U = alg.update_U()
    alg.V_old = alg.V
    alg.V = alg.update_V()

    pred_y = np.argmax(alg.U, axis=1)

    train_accuracy = accuracy_score(train_y, pred_y[:train_size])
    test_accuracy = accuracy_score(test_y, pred_y[train_size:])
    print("Iteration: ", l, "Train Accuracy: ",
          train_accuracy, "Test Accuracy: ", test_accuracy)

    if alg.is_converged():
        break
