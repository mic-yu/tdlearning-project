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

path_gt = "./data/mueller_q_beta_0p1.csv"
df = pd.read_csv(path_gt, header=None)     # each row: x,y
print(df)
qlist = df.iloc[0].values
print(len(qlist))



qlist = np.array(qlist)
qlist = qlist * (-1) + 1


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

plt.scatter(pts[:, 0], pts[:, 1], c=qlist[1:], cmap='viridis')
plt.colorbar(label="probability")
