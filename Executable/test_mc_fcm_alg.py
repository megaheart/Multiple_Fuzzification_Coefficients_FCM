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
from Components.Algorithms.MC_FCM import MC_FCM
import Components.Cores.clusters_generations as cg
import Components.Cores.helper_fns as hepler

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
alg = MC_FCM()
# prepare data
X = elements
Y = label

# %%
# initialize variables; initialize U, V, m2
print("Initializing variables")
N = X.shape[0]
alg.X = X
alg.C = C
alg.U = np.zeros((N, C))
alg.epsilon = 0.0001
max_iter = 200
print("Initializing cluster centers and m")
alg.m, alg.V = alg.generate_m_and_first_V(ml = 1.1, mu=4.1)
print(alg.m)

X_same = np.copy(X)
Y_same = np.copy(Y)
pred_y = np.full(N, -1)
print("Start solving the algorithm")
for l in range(max_iter):
    alg.U = alg.update_U()
    alg.V_old = alg.V
    alg.V = alg.update_V()

    pred_y = np.argmax(alg.U, axis=1)
    pred_y = hepler.sync_output_for_nonsupervised_learning(pred_y, Y)

    accuracy = accuracy_score(Y, pred_y)
    print("Iteration: ", l, "Accuracy: ", accuracy)

    if alg.is_converged():
        break

print("Same accuracy: ", np.count_nonzero(pred_y == Y) / N)

# Check whether X == X_same and Y == Y_same
print("X == X_same: ", np.array_equal(X, X_same))
print("Y == Y_same: ", np.array_equal(Y, Y_same))