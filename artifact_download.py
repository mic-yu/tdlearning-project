import os
import wandb 
import torch

from GoalNet import GoalNet

api = wandb.Api()
artifact = api.artifact("dpinchuk-university-of-wisconsin-madison/test_sweep_2/2025-08-22_13-49-17_5whgz1n6_final_model:v0")
directory = "./artifact_download_test/"
os.makedirs(directory, exist_ok=True)
path = artifact.download(root=directory)

model_path = os.path.join(path, "test_4_trial_2025-08-22_13-49-17.pth")


run_path = "dpinchuk-university-of-wisconsin-madison/test_sweep_2/5whgz1n6"
run = api.run(run_path)

input_size = 9
hidden_sizes = []
n_layers = run.config["params"]["n_layers"]
for i in range(n_layers):
    hidden_sizes.append(run.config[f"neurons_layer_{i}"])

model = GoalNet(input_size, hidden_sizes)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
with torch.no_grad():
    y = model(torch.tensor([1,2,3,4,5,6,7,8,9]) / 9)
print("y: ", y)
print("complete")
#print("path: ", path)