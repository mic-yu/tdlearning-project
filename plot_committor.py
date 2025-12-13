import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
import os

from artifact_download import load_wandb_model

path_gt = "./data/mueller_TPTdata_beta_0p1_2025.csv"
df = pd.read_csv(path_gt, header=None)     # each row: x,y
data = df.to_numpy()
pts = data[:, :2]
q_gt = data[:, 2]
q_gt = q_gt * (-1) + 1



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "./artifact_download_muller/ep_400_l5_BCE_trial_2025-12-12_06-37-06_mvjlar04_best.pth"
run_path = "dpinchuk-university-of-wisconsin-madison/td_committor_ep400/mvjlar04"
assert os.path.exists(model_path)
model = load_wandb_model(run_path, model_path, device, input_size=2)
model.eval()

pts = torch.from_numpy(pts).to(dtype=torch.float32)
output = model(pts)

print("inference complete")
print("output.shape:", output.shape)

q_gt = torch.from_numpy(q_gt).to(dtype=torch.float32)
mse_loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()
print("MSE: ", mse_loss(output.squeeze(), q_gt.squeeze()))
print("L1: ", l1_loss(output.squeeze(), q_gt.squeeze()))
max_l1 = torch.max(torch.abs(output.squeeze() - q_gt.squeeze()))
print("max L1: ", max_l1)

# out_list = []
# for i in range(len(output)):
#     out_list.append(output.squeeze().detach()[i].item())
output = output.squeeze().numpy(force=True)
pts = pts.numpy(force=True)
print(type(output))
print(output.shape)
print(output.dtype)

plt.rcParams['figure.dpi'] = 600
plt.scatter(pts[:, 0], pts[:, 1], c=output, cmap='viridis')
plt.colorbar(label='probability')