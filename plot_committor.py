import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
import os

from artifact_download import load_wandb_model

path = "./data/mueller_pts.csv"
df = pd.read_csv(path)     # each row: x,y
pts = torch.tensor(df.values, dtype=torch.float32)

print(pts.shape)   # (N, 2)
#plt.scatter(pts[:, 0], pts[:, 1])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_path = "./artifact_download/tdgoal_h100_ep500_epoch200_l5_trial_2025-08-27_16-10-28_pv5kiybp_best.pth"
# run_path = "dpinchuk-university-of-wisconsin-madison/tdgoal_h100_ep500_epoch200/pv5kiybp"
model_path = "./artifact_download_muller/ep_100_l5_BCE_trial_2025-12-08_22-38-16_m5zv1tke_best.pth"
run_path = "dpinchuk-university-of-wisconsin-madison/td_committor_ep100/m5zv1tke"
assert os.path.exists(model_path)
model = load_wandb_model(run_path, model_path, device)
model.eval()

output = model(pts)

print("inference complete")

# plt.scatter(pts[:, 0], pts[:, 1], c=output.numpy(force=True), cmap='viridis')
# plt.colorbar(label="probability")

path_gt = "./data/mueller_q_beta_0p1.csv"
df = pd.read_csv(path_gt, header=None)     # each row: x,y
print(df)
qlist = df.iloc[0].values
print(len(qlist))

qlist = np.array(qlist)
qlist = qlist * (-1) + 1
q = torch.from_numpy(qlist)


mse_loss = torch.nn.MSELoss()
print(mse_loss(output.squeeze(), q[1:].squeeze()))