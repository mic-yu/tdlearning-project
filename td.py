import numpy as np
import hydra
import optuna
import pickle
import torch
import math
import os
import wandb
import copy
import time

from torch.utils.data import TensorDataset, random_split, DataLoader
from optuna.integration.wandb import WeightsAndBiasesCallback
from datetime import datetime
from functools import partial
from omegaconf import OmegaConf
from tqdm import tqdm
from torch import nn

from utils import load_data_tensors, get_ba_from_conf
from GoalNet import GoalNet
from train import train, eval


def get_episode_data(path, train_val_test_split):
    with open(path, 'rb') as f:
        obsList = pickle.load(f)

    #turn each episode into a single numpy array
    for i in range(len(obsList)):
        obsList[i] = torch.tensor(np.vstack(obsList[i]), dtype=torch.float32)


    #do train/val/test split by episodes
    num_eps = len(obsList)
    num_val = math.floor(train_val_test_split[1] * num_eps)
    num_test = math.ceil(train_val_test_split[2] * num_eps)
    num_train = num_eps - num_test - num_val

    train_list = obsList[:num_train]
    val_list = obsList[num_train: num_train + num_val]
    test_list = obsList[num_train + num_val:]

    return train_list, val_list, test_list

def get_td_train_data(data_list, cfg):
    """
    Args:
        data (List of episodes)
    """
    max_k = cfg.max_k
    new_ep_list = []
    for ep in data_list:
        tile_length = ep.size(dim=0) - 1
        #print("ep.size: ", ep.size())
        for k in range(2, max_k+1):           
            h_feat = torch.tensor(k).tile((tile_length, 1)) #horizon feature
            terminal_info = torch.tensor(0).tile((tile_length, 1))
            terminal_info[-1][0] = 1
            new_sample = torch.cat((ep[:-1, :-1], ep[1:, :-1], h_feat, terminal_info, ep[1:, -1:]), dim=1)    
            new_sample[-k:, -1] = ep[-1][-1]
            #print("new_sample.size: ", new_sample.size())
            #print("tile size: ", tile_length)
            new_ep_list.append(new_sample)

        #now do k=1
        last_h_feat = torch.tensor(1).tile((tile_length, 1))
        terminal_info = torch.tensor(1).tile((tile_length, 1))
        last_k_sample = torch.cat((ep[:-1, :-1], ep[1:, :-1], last_h_feat, terminal_info, ep[1:, -1:]), dim=1)
        #print("last_k_sample.size: ", last_k_sample.size())
        new_ep_list.append(last_k_sample)
        
        
    dataTensor = torch.cat(new_ep_list, dim=0)
    return dataTensor

def td_data_update(model, data_list, cfg):
    """
    Args:
        data (List of episodes)
    """
    model.eval()
    max_k = cfg.max_k
    new_ep_list = []
    for ep in data_list:
        tile_length = ep.size(dim=0) - 2
        for k in range(2, max_k+1):           
            h_feat = torch.tensor(k / max_k).tile((tile_length, 1)) #horizon feature
            h_feat_next = torch.tensor((k-1) / max_k).tile((tile_length, 1)) #next horizon feature
            #print("h_feat.size: ", h_feat.size())
            #print("ep.size: ", ep.size())
            boot_v = model(torch.cat((ep[1:-1, :-1], h_feat_next), dim=1))
            new_feature_data = torch.cat((ep[:-2, :-1], h_feat), dim=1)
            new_data = torch.cat((new_feature_data, boot_v), dim=1)
            
            new_terminal_sample = torch.cat((ep[-2, :-1], torch.tensor([k / max_k]), ep[-1, -1:])).unsqueeze(0)
            #print("new_sample size: ", new_terminal_sample.size())
            new_data = torch.cat((new_data, new_terminal_sample), dim=0)
            #print("new data size: ", new_data.size())
            new_ep_list.append(new_data)

        #now do k=1
        last_h_feat = torch.tensor(1 / max_k).tile((tile_length + 1, 1))
        last_k_sample = torch.cat((ep[:-1, :-1], last_h_feat, ep[1:, -1:]), dim=1)
        new_ep_list.append(last_k_sample)
        
    dataTensor = torch.cat(new_ep_list, dim=0)
    return dataTensor

def get_val_data(data_list, cfg):
    """
    Args:
        data (List of episodes)
    """
    max_k = cfg.max_k
    new_ep_list = []
    for ep in data_list:
        tile_length = ep.size(dim=0) - 1
        for k in range(1, max_k+1):  

            #add k feature
            h_feat = torch.tensor(k / max_k).tile((tile_length, 1)) #horizon feature
            new_feature_data = torch.cat((ep[:-1, :-1], h_feat, ep[1:, -1:]), dim=1)
            #label the data according to horizon k
            new_feature_data[-k:, -1] = ep[-1][-1]
            
            new_ep_list.append(new_feature_data)

        #now do k=1
        # last_h_feat = torch.tensor(1 / max_k).tile((tile_length, 1))
        # last_k_sample = torch.cat((ep[:-1, :-1], last_h_feat, ep[1:, -1:]), dim=1)
        # new_ep_list.append(last_k_sample)
        
    dataTensor = torch.cat(new_ep_list, dim=0)
    return dataTensor
    

def train_td(trial, model, trainData, valDataLoader, cfg, params):
    for epoch in range(cfg.num_epochs):
        with torch.no_grad():
            dataTensor = td_data_update(model, trainData, cfg)
        nn_dataset = TensorDataset(dataTensor[:, :-1], dataTensor[:, -1:])
        dataloader = DataLoader(nn_dataset, batch_size=params["bs"], shuffle=True)
        model = train(trial, model, dataloader, cfg, params)

        if (epoch + 1) % cfg.eval_frequency == 0:
            val_loss, conf_mx = eval(model, valDataLoader, params["loss_fn"])
            trial.report(-val_loss, epoch)
            BA = get_ba_from_conf(*conf_mx.ravel())
            wandb_report = {"epoch": epoch,
                            "val_loss": val_loss,
                            "BA": BA,
                            "len_train_dataloader": len(dataloader)
                            }
            wandb.log(wandb_report)
    return model

def get_target_from_batch(batch, targetModel, cfg, params):
    """
    batch (Tensor): N x ((state s ), (state s'), k (for s), terminal_info (for s'), true_value)
    """
    assert (batch.size(dim=1) - 3) % 2 == 0
    state_dim = int((batch.size(dim=1) - 3) / 2)
    max_k = cfg.max_k
    with torch.no_grad():
        target_input = batch[:, state_dim:-2]
        target_input[:, -1:] = target_input[:, -1:] - 1
        target_input[:, -1:] = target_input[:, -1:] / max_k
        Y = targetModel(target_input)
    
    for idx in range(batch.size(dim=0)):
        if batch[idx][-2] == 1:
            Y[idx][0] = batch[idx][-1]
    
    X = torch.cat((batch[:, :state_dim], batch[:, -3:-2]), dim=1)
    return X, Y

        
    

def train_td_new(trial, wb_run, model, trainDataList, valDataLoader, cfg, params):
    trainData = get_td_train_data(trainDataList, cfg)
    trainData = trainData.to(device=params["device"])
    trainDataset = TensorDataset(trainData)
    #generator = torch.Generator().manual_seed(37)
    #debugDataset, _ = torch.utils.data.random_split(trainDataset, [0.2, 0.8], generator=generator)
    trainDataLoader = DataLoader(trainDataset, batch_size=params["bs"], shuffle=True)

    lr = params["lr"]
    loss_fn = params["loss_fn"]
    num_epochs = cfg.num_epochs
    if num_epochs == 0:
        num_epochs = 1
    assert num_epochs > 0


    #Data
    model_path = params["model_storage_path"] #for saving
    best_model_path = params["best_model_storage_path"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #min_loss = float('inf')
    #patience = 4 #times we can have an evaluation that is worse than minimum
                 # corresponds to patience*5 epochs
    #patience_counter = 0
    #eps = 1e-5 #by how much loss should improve over patience period
    sig = nn.Sigmoid()

    #Training loop
    step = 0
    best_ba = 0.0
    targetModel = copy.deepcopy(model)
    targetModel.eval()
    targetModel.to(device=params["device"])
    start_time = time.time()
    for epoch in range(num_epochs): #)position=0, leave=False):
        running_avg = 0.0
        running_count = 0
        model.train()
        for batch, in trainDataLoader:
            X, Y = get_target_from_batch(batch, targetModel, cfg, params)
            optimizer.zero_grad()
            outputs = model(X)
            if cfg.train_forward_sig:
                outputs = sig(outputs)

            loss = loss_fn(outputs, Y)
            #print(f"Step: {step} Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            
            running_avg = running_avg*running_count + loss.item()*X.size(dim=0)
            running_count = running_count + X.size(dim=0)
            running_avg = running_avg/running_count

            step = step + 1
            if (step + 1) % cfg.train_loss_frequency == 0:
                #print(f"Epoch: {epoch} Step: {step} Loss: {running_avg}")
                wb_run.log({"train/loss": running_avg,
                           "optimization step": step,
                           })
                running_avg = 0.0
                running_count = 0
            if step % params["target_update_frequency"] == 0:
                targetModel = copy.deepcopy(model)
                targetModel.eval()
                targetModel.to(device=params["device"])
        if (epoch + 1) % cfg.eval_frequency == 0:
            val_loss, conf_mx = eval(model, valDataLoader, params["loss_fn"])
            if trial is not None:
                trial.report(-val_loss, epoch)
            BA = get_ba_from_conf(*conf_mx.ravel())
            elapsed_time = (time.time() - start_time) / 60
            wandb_report = {"epoch": epoch,
                            "optimization step": step,
                            "train/loss": running_avg,
                            "validation/loss": val_loss,
                            "validation/Balanced Accuracy": BA,
                            "Elapsed Time Minutes": elapsed_time
                            }
            wb_run.log(wandb_report)
            if BA >= best_ba:
                torch.save(model.state_dict(), best_model_path)
                wb_run.summary["Best BA"] = BA
                wb_run.summary["Best Epoch"] = epoch
                best_ba = BA
                

            
        #print("Epoch {} complete with avg train loss {}".format(epoch, running_avg))
        #train_loss_list.append(running_avg)

        #Evaluation loop and early stopping
        '''
        if (epoch + 1) % 5 == 0:
            val_loss, _ = eval(model, valDataLoader, loss_fn)
            trial.report(-val_loss, epoch)
            print("Epoch {} complete with avg val loss {}".format(epoch, val_loss))
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(model.state_dict(), model_path)
                patience_counter = 0
            if trial.should_prune():
                #trial.set_user_attr("pruned", True)
                model.load_state_dict(torch.load(model_path, weights_only=True))
                _, conf_mx = eval(model, valDataLoader, loss_fn)
                BA = get_ba_from_conf(*conf_mx.ravel())
                trial.report(BA, epoch + 1)
                trial.set_user_attr("epochs_trained", epoch + 1)
                raise optuna.TrialPruned()
            else:
                patience_counter += 1
                if patience_counter > patience:
                    trial.set_user_attr("epochs_trained", epoch + 1)
                    model.load_state_dict(torch.load(model_path, weights_only=True))
                    return model 
        '''
    '''
    trial.set_user_attr("epochs_trained", epoch+1)
    '''

    #model_path = model_storage_path + "/{}_trial_{}".format(study_name, trial.number)
    torch.save(model.state_dict(), model_path)
    return model


def single_objective(cfg, trial=None):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"trial_{trial.number}_{current_time}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_storage_dir = cfg.optuna_settings.model_storage_dir
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

    loss_fn = nn.MSELoss()
    #loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
    best_ba = wb_run.summary["Best BA"]
    wb_run.finish()
    
    return best_ba

def objective(trial, cfg):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"trial_{trial.number}_{current_time}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_storage_dir = cfg.optuna_settings.model_storage_dir
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

    loss_fn = nn.MSELoss()
    #loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
    best_ba = wb_run.summary["Best BA"]
    wb_run.finish()
    
    return best_ba

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
    wandb.login()
    if cfg.mode == "sweep":
        partial_objective = partial(objective,  cfg=cfg)
    elif cfg.mode == "single_run":
        partial_objective = partial(single_objective, cfg=cfg)
    elif cfg.mode == "wandb_sweep":
        partial_function = partial(single_objective, cfg=cfg)
        wandb.agent(sweep_id = cfg.wandb_sweep.sweep_id, function=partial_function, count=1)
    else:
        message = f"cfg.mode {cfg.mode} is not implemented."
        raise ValueError(message)

    #wandbc_decorator = wandbc.track_in_wandb()
    #decorated_objective = wandbc_decorator(partial_objective)

    study_name = cfg.optuna_settings.study_name
    model_storage_dir = cfg.optuna_settings.model_storage_dir
    num_trials = cfg.optuna_settings.num_trials
    db_dir = cfg.optuna_settings.db_dir
    os.makedirs(db_dir, exist_ok=True)
    db_name = cfg.optuna_settings.db_name
    db_path = os.path.join("sqlite:///", db_dir, db_name)
    print("db_path: ", db_path)

    os.makedirs(model_storage_dir, exist_ok=True)
    n_startup_trials = cfg.optuna_settings.n_startup_trials
    study = optuna.create_study(direction="maximize",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=50, interval_steps=1), 
                study_name=study_name, 
                storage=db_path, 
                load_if_exists=True)
    study.optimize(partial_objective, n_trials=num_trials, n_jobs=1) # callbacks=[wandbc])


if __name__ == '__main__':
    run()
