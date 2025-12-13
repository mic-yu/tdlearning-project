import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
import os



path_gt = "./data/mueller_TPTdata_beta_0p1_2025.csv"
#path_gt = "./data/mueller_q_beta_0p1.csv"
df = pd.read_csv(path_gt, header=None)     # each row: x,y

data = df.to_numpy()
pts = data[:, :2]
q = data[:, 2]

q = q * (-1) + 1

print(type(q))
print(q.shape)
print(q.dtype)
print(type(pts))
quit()

plt.rcParams['figure.dpi'] = 600
plt.scatter(pts[:, 0], pts[:, 1], c=q, cmap='viridis')
plt.colorbar(label="probability")
