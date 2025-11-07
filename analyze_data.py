from matplotlib import pyplot as plt
import numpy as np

from utils import load_data_tensors


path = "./data/n_eps-100-env-base_agent_env_2025-11-06_21-07-54.pkl"


datasets, lastObs = load_data_tensors(path, [0.6, 0.2, 0.2], None)
print(len(datasets))
lastObs = np.array(lastObs)
print("lastObs:", lastObs.shape)

trainData = datasets[0]

print(trainData.size())
print(lastObs)
#plt.scatter(trainData[:, 2], trainData[:, 3], s=1)
fig = plt.figure(figsize=(15, 10), dpi=300)
plt.xlim(-4500, 4500)
plt.ylim(-3000, 3000)
plt.scatter(lastObs[:, 2], lastObs[:, 3], c=lastObs[:, -1])
plt.colorbar()
plt.show()
