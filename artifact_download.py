import os
import wandb 
import torch

from GoalNet import GoalNet

def download_model(artifact_name, download_folder):
    """

    """
    api = wandb.Api()
    artifact = api.artifact(artifact_name)
    os.makedirs(download_folder, exist_ok=True)
    artifact.download(root=download_folder)


def load_wandb_model(run_path, model_path, device):
    run = wandb.Api().run(run_path)

    input_size = 6
    hidden_sizes = []
    n_layers = run.config["params"]["n_layers"]
    for i in range(n_layers):
        hidden_sizes.append(run.config[f"neurons_layer_{i}"])

    model = GoalNet(input_size, hidden_sizes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model


if __name__ == '__main__':
    # artifact_name = "dpinchuk-university-of-wisconsin-madison/tdgoal_h100_ep500_epoch200/2025-08-27_16-10-28_pv5kiybp_best_model:v0"
    #download_folder = "./artifact_download_muller/"
    download_folder = "./artifact_download/"
    #artifact_name = "dpinchuk-university-of-wisconsin-madison/td_abs/2025-11-07_20-40-04_uydoh1wg_best_model:v0"
    #artifact_name = "dpinchuk-university-of-wisconsin-madison/td_abs_inf_500ep/2025-12-01_03-58-56_ssc8vz8m_best_model:v0"
    artifact_name = "dpinchuk-university-of-wisconsin-madison/td_abs_inf_500ep/2025-12-11_22-15-31_yy01it4b_best_model:v0"
    download_model(artifact_name, download_folder)