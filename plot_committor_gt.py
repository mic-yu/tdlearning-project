import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
import os

path = "./data/mueller_pts.csv"
df = pd.read_csv(path)     # each row: x,y
pts = torch.tensor(df.values, dtype=torch.float32)

print(pts.shape)   # (N, 2)
#plt.scatter(pts[:, 0], pts[:, 1])

path_gt = "./data/mueller_TPTdata_beta_0p1_2025.csv"
path_gt = "./data/mueller_q_beta_0p1.csv"
df = pd.read_csv(path_gt, header=None)     # each row: x,y
print(df)
qlist = df.iloc[0].values
#print(len(qlist))
data = df.to_numpy()
#pts = data[:, :2]
q = data[:, 2]

data_rare = []
for i in range(data.shape[0]):
    if i % 10 == 0:
        data_rare.append([data[i][0], data[i][1]])

print('rare')
data_rare = np.array(data_rare)
print(data_rare)
print(data_rare.shape)


#qlist = np.array(qlist)
q = qlist
q = q * (-1) + 1
print(q)
print(pts)

'''
clean_list = []
for x in qlist:
    if x == '0.0.1':
        continue
    else:
        clean_list.append(float(x))
print("clean: ", len(clean_list))
print("pts: ", pts.size())
'''

#print(q_gt)

plt.scatter(pts[:, 0], pts[:, 1], c=q[1:], cmap='viridis')
#plt.scatter(data_rare[:, 0], data_rare[:, 1])
plt.colorbar(label="probability")
