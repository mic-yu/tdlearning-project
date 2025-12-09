import numpy as np
import hydra
import optuna
import torch
import os
import wandb
import random

from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from functools import partial
from omegaconf import OmegaConf
from torch import nn

from utils import ( get_ba_from_conf, 
                    load_abs_datasets,
                    get_abs_episode_data)
from GoalNet import GoalNet
from train_td import train_td, eval


def set_seed(seed=37):
    """
    Set random seeds for reproducibility across all libraries
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def wandb_sweep_objective(hydra_cfg):
    """
    Objective for wandb sweep to optimize
    
    Args: 
        hydra_cfg: hydra config containing params

    Returns: 
        float: value of objective (best balanced accuracy of over training)
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wb_config = OmegaConf.to_container(hydra_cfg) #do I still need this?
    wandb_kwargs = {
        "entity": hydra_cfg.wandb.entity,
        "project": hydra_cfg.wandb.project,
        }
    wb_run = wandb.init(**wandb_kwargs)
    wb_run.name = current_time + "_" + wb_run.id
    run_name = wb_run.name


    param_config = wandb.config 
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_storage_dir = hydra_cfg.model_storage_dir
    sweep_name = hydra_cfg.wandb_sweep.name
    if hydra_cfg.chtc == False:
        model_storage_dir = os.path.join(model_storage_dir, sweep_name)
    os.makedirs(model_storage_dir, exist_ok=True)
    model_storage_path = os.path.join(model_storage_dir, "{}_trial_{}.pth".format(sweep_name, run_name))
    best_model_storage_path = os.path.join(model_storage_dir, "{}_trial_{}_best.pth".format(sweep_name, run_name))

    #initialize hyperparameters
    n_layers = hydra_cfg.wandb_sweep.n_layers
    hidden_sizes = []
    for i in range(n_layers):
        hidden_sizes.append(param_config[f"neurons_layer_{i}"])
    lr = param_config["lr"]
    batch_size = param_config["bs"]  
    target_update_frequency = param_config["target_update_frequency"]


    if hydra_cfg.loss_fn == "MSE":
        loss_fn = nn.MSELoss()
    elif hydra_cfg.loss_fn == "BCE":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = None
        message = f"cfg.loss_fn {hydra_cfg.loss_fn} is not implemented."
        raise ValueError(message)

    params = {"n_layers": n_layers, 
              "hidden_sizes": hidden_sizes, 
              "lr": lr, 
              "bs": batch_size,
              "loss_fn": loss_fn,
              "model_storage_path": model_storage_path,
              "best_model_storage_path": best_model_storage_path,
              "target_update_frequency": target_update_frequency
              }

    #Data
    trainList, _, _ = get_abs_episode_data(hydra_cfg.path, hydra_cfg.train_val_test_split, ep_np=False, preprocess=False)
    #valData = get_val_data(valList, hydra_cfg).to(device=device)
    #valDataset = TensorDataset(valData[:, :-1], valData[:, -1:])
    _, valDataset, _ = load_abs_datasets(hydra_cfg.path, hydra_cfg.train_val_test_split, -1, device=device, preprocess=False, )
    generator = torch.Generator().manual_seed(37)
    valDataLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True, generator=generator)

    #training
    input_size = trainList[0].size(dim=1) - 1
    myModel = GoalNet(input_size, hidden_sizes).to(device=device)

    params["input_size"] = input_size
    
    wb_run.config.update({"hydra_config": OmegaConf.to_container(hydra_cfg), "params": params})

    params["device"] = device
    trainedModel = train_td(None, wb_run, myModel, trainList, valDataLoader, hydra_cfg, params)
    final_loss, conf_mx = eval(trainedModel, valDataLoader, loss_fn)
    BA = get_ba_from_conf(*conf_mx.ravel())

    wb_run.summary["Final BA"] = BA
    wb_run.summary["Final eval loss"] = final_loss
    best_val_ba = wb_run.summary["Best BA"]
    wb_run.finish()
    
    return best_val_ba


def single_objective(cfg, trial=None):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"trial_{trial.number}_{current_time}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_storage_dir = cfg.model_storage_dir
    study_name = cfg.optuna_settings.study_name
    model_storage_dir = os.path.join(model_storage_dir, study_name)
    os.makedirs(model_storage_dir, exist_ok=True)
    model_storage_path = os.path.join(model_storage_dir, "{}_trial_{}.pth".format(study_name, trial.number))
    best_model_storage_path = os.path.join(model_storage_dir, "{}_trial_{}_best.pth".format(study_name, trial.number))

    #initialize hyperparameters
    n_layers = cfg.single_run.n_layers
    hidden_sizes = cfg.single_run.hidden_sizes
    lr = cfg.single_run.lr
    batch_size = cfg.single_run.bs    
    target_update_frequency = cfg.single_run.target_update_frequency

    #log hyperparameters to optuna.
    if trial is not None:
        trial.suggest_int('n_layers', n_layers, n_layers)
        for i in range(n_layers):
            trial.suggest_int('neurons_layer_{}'.format(i), hidden_sizes[i], hidden_sizes[i])
        trial.suggest_float('lr', lr, lr)
        trial.suggest_int('bs', batch_size, batch_size)
        trial.suggest_int('target_update_frequency', target_update_frequency , target_update_frequency)

    if cfg.loss_fn == "MSE":
        loss_fn = nn.MSELoss()
    elif cfg.loss_fn == "BCE":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = None
        message = f"cfg.loss_fn {cfg.loss_fn} is not implemented."
        raise ValueError(message)

    params = {"n_layers": n_layers, 
              "hidden_sizes": hidden_sizes, 
              "lr": lr, 
              "bs": batch_size,
              "loss_fn": loss_fn,
              "model_storage_path": model_storage_path,
              "best_model_storage_path": best_model_storage_path,
              "target_update_frequency": target_update_frequency
              }

    #Data
    trainList, valList, _ = get_episode_data(cfg.path, cfg.train_val_test_split)
    valData = get_val_data(valList, cfg).to(device=device)
    valDataset = TensorDataset(valData[:, :-1], valData[:, -1:])
    valDataLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)

    #training
    input_size = trainList[0].size(dim=1)
    myModel = GoalNet(input_size, hidden_sizes).to(device=device)

    params["input_size"] = input_size
    params["trial_number"] = trial.number

    wb_config = OmegaConf.to_container(cfg)
    wb_config["params"] = params
    wandb_kwargs = {
        "entity": cfg.wandb.entity,
        "project": cfg.wandb.project,
        "config": wb_config,
        "name": run_name
        }
    wb_run = wandb.init(**wandb_kwargs)
    #trainedModel = train_td(trial, myModel, trainList, valDataLoader, cfg, params)
    params["device"] = device
    trainedModel = train_td_new(trial, wb_run, myModel, trainList, valDataLoader, cfg, params)
    final_loss, conf_mx = eval(trainedModel, valDataLoader, loss_fn)
    BA = get_ba_from_conf(*conf_mx.ravel())

    wb_run.summary["Final BA"] = BA
    wb_run.summary["Final eval loss"] = final_loss
    best_val_ba = wb_run.summary["Best BA"]
    wb_run.finish()
    
    return best_val_ba

def objective(trial, cfg):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"trial_{trial.number}_{current_time}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_storage_dir = cfg.model_storage_dir
    study_name = cfg.optuna_settings.study_name
    model_storage_dir = os.path.join(model_storage_dir, study_name)
    os.makedirs(model_storage_dir, exist_ok=True)
    model_storage_path = os.path.join(model_storage_dir, "{}_trial_{}.pth".format(study_name, trial.number))
    best_model_storage_path = os.path.join(model_storage_dir, "{}_trial_{}_best.pth".format(study_name, trial.number))

    #initialize hyperparameters with optuna from config ranges
    n_layers_min = cfg.search_space.n_layers_range[0]
    n_layers_max = cfg.search_space.n_layers_range[1]
    n_layers = trial.suggest_int('n_layers', n_layers_min, n_layers_max)

    num_neurons_min = cfg.search_space.num_neurons_range[0]
    num_neurons_max = cfg.search_space.num_neurons_range[1]
    hidden_sizes = []
    for i in range(n_layers):
        num_neurons = trial.suggest_int('neurons_layer_{}'.format(i), num_neurons_min, num_neurons_max)
        hidden_sizes.append(num_neurons)

    lr_min = cfg.search_space.lr_range[0]
    lr_max = cfg.search_space.lr_range[1]
    lr = trial.suggest_float('lr', lr_min, lr_max, log=True)

    bs_min = cfg.search_space.bs_range[0]
    bs_max = cfg.search_space.bs_range[1]
    bs_step = cfg.search_space.bs_step
    batch_size = trial.suggest_int('bs', bs_min, bs_max, step=bs_step)
    
    tuf_min = cfg.search_space.target_update_frequency_range[0]
    tuf_max = cfg.search_space.target_update_frequency_range[1]
    tuf_step = cfg.search_space.tuf_step
    target_update_frequency = trial.suggest_int('target_update_frequency', tuf_min, tuf_max, step=tuf_step)

    if cfg.loss_fn == "MSE":
        loss_fn = nn.MSELoss()
    elif cfg.loss_fn == "BCE":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = None
        message = f"cfg.loss_fn {cfg.loss_fn} is not implemented."
        raise ValueError(message)

    params = {"n_layers": n_layers, 
              "hidden_sizes": hidden_sizes, 
              "lr": lr, 
              "bs": batch_size,
              "loss_fn": loss_fn,
              "model_storage_path": model_storage_path,
              "best_model_storage_path": best_model_storage_path,
              "target_update_frequency": target_update_frequency
              }

    #Data
    trainList, valList, _ = get_episode_data(cfg.path, cfg.train_val_test_split)
    valData = get_val_data(valList, cfg).to(device=device)
    valDataset = TensorDataset(valData[:, :-1], valData[:, -1:])
    valDataLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)

    #training
    input_size = trainList[0].size(dim=1)
    myModel = GoalNet(input_size, hidden_sizes).to(device=device)

    params["input_size"] = input_size
    params["trial_number"] = trial.number

    wb_config = OmegaConf.to_container(cfg)
    wb_config["params"] = params
    wandb_kwargs = {
        "entity": cfg.wandb.entity,
        "project": cfg.wandb.project,
        "config": wb_config,
        "name": run_name
        }
    wb_run = wandb.init(**wandb_kwargs)
    #trainedModel = train_td(trial, myModel, trainList, valDataLoader, cfg, params)
    params["device"] = device
    trainedModel = train_td_new(trial, wb_run, myModel, trainList, valDataLoader, cfg, params)
    final_loss, conf_mx = eval(trainedModel, valDataLoader, loss_fn)
    BA = get_ba_from_conf(*conf_mx.ravel())

    wb_run.summary["Final BA"] = BA
    wb_run.summary["Final eval loss"] = final_loss
    best_val_ba = wb_run.summary["Best BA"]
    wb_run.finish()
    
    return best_val_ba

@hydra.main(version_base=None, config_path=".", config_name="config")
def run(cfg):
    # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # run_name = current_time
    # wandb_kwargs = {
    #     "entity": cfg.wandb.entity,
    #     "project": cfg.wandb.project,
    #     "config": OmegaConf.to_container(cfg),
    #     "name": run_name
    #     }
    #wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)
    set_seed(37)
    wandb.login()
    if cfg.mode == "sweep":
        partial_objective = partial(objective,  cfg=cfg)
    elif cfg.mode == "single_run":
        partial_objective = partial(single_objective, cfg=cfg)
    elif cfg.mode == "wandb_sweep":
        partial_function = partial(wandb_sweep_objective, hydra_cfg=cfg)
        wandb.agent(sweep_id = cfg.wandb_sweep.sweep_id, 
                    function=partial_function,
                    entity=cfg.wandb.entity,
                    project=cfg.wandb.project,
                    count=cfg.wandb_sweep.count)
    else:
        message = f"cfg.mode {cfg.mode} is not implemented."
        raise ValueError(message)

    #wandbc_decorator = wandbc.track_in_wandb()
    #decorated_objective = wandbc_decorator(partial_objective)

    model_storage_dir = cfg.model_storage_dir
    os.makedirs(model_storage_dir, exist_ok=True)
    
    if cfg.mode != "wandb_sweep":
        study_name = cfg.optuna_settings.study_name
        num_trials = cfg.optuna_settings.num_trials
        db_dir = cfg.optuna_settings.db_dir
        os.makedirs(db_dir, exist_ok=True)
        db_name = cfg.optuna_settings.db_name
        db_path = os.path.join("sqlite:///", db_dir, db_name)
        print("db_path: ", db_path)

        
        n_startup_trials = cfg.optuna_settings.n_startup_trials
        study = optuna.create_study(direction="maximize",
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=50, interval_steps=1), 
                    study_name=study_name, 
                    storage=db_path, 
                    load_if_exists=True)
        study.optimize(partial_objective, n_trials=num_trials, n_jobs=1) # callbacks=[wandbc])


if __name__ == '__main__':
    run()